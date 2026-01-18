import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import color, data, draw, feature, img_as_float
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
def test_hog_orientations_circle():
    width = height = 100
    image = np.zeros((height, width))
    rr, cc = draw.disk((int(height / 2), int(width / 2)), int(width / 3))
    image[rr, cc] = 100
    image = filters.gaussian(image, sigma=2, mode='reflect')
    for orientations in range(2, 15):
        hog, hog_img = feature.hog(image, orientations=orientations, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, transform_sqrt=False, block_norm='L1')
        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.colorbar()
            plt.title('image_float')
            plt.subplot(1, 2, 2)
            plt.imshow(hog_img)
            plt.colorbar()
            plt.title(f'HOG result visualisation, orientations={orientations}')
            plt.show()
        hog_matrix = hog.reshape(-1, orientations)
        actual = np.mean(hog_matrix, axis=0)
        desired = np.mean(hog_matrix)
        assert_almost_equal(actual, desired, decimal=1)