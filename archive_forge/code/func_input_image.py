import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.ndimage import correlate
from skimage import draw
from skimage._shared.testing import fetch
from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize, thin
from skimage.morphology._skeletonize import G123_LUT, G123P_LUT, _generate_thin_luts
@property
def input_image(self):
    """image to test thinning with"""
    ii = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 0], [0, 1, 0, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 6, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=float)
    return ii