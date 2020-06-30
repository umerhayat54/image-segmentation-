import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r"data.jpg", 1)

# kernel is normaly is square and like same shape which we want to apply on the image
# in case kernel is 3 by 3 square shape apply on our image whenever these black dots are there
kernal = np.ones((3, 3), np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#  noramly we prerfrom  Morphological transformation on binary  images
#thats why we to provide a marks to a image using simple thresholding


# Background area using Dialation
#reduce black dot
dilation = cv2.dilate(img, kernal, iterations=2)


# while erosion removes pixels on object boundaries.
erosion = cv2.erode(img, kernal, iterations=1)

#Dilation and erosion are often used in combination to implement image processing operations. ...
# Morphological opening is useful for removing small objects from an image
# while preserving the shape and size of larger objects in the image.
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernal)

#Closing is reverse of Opening, Dilation followed by Erosion.
# It is useful in closing small holes inside the foreground objects, or small black points on the object
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernal)

#Morphological Gradient
#It is the difference between dilation and erosion of an image.
mg = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernal)

#Top Hat
#It is the difference between input image and Opening of the image. Below example is done for a 9x9 kernel.
th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernal)


titles = ['original image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'Morphological Gradient', 'Top Hat']
images = [img, img, dilation, erosion, opening, closing, mg, th]

for i in range(8):
    #plt for creates a plotting area in a figure, plots some lines in a plotting area,
    plt.subplot(2, 4, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
#waitkey use for stay windowframe
cv2.waitKey(0)
#destroyAllWindows use for finished the windowframe
cv2.destroyAllWindows()