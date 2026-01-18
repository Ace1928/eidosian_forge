import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage import data, color, morphology
from skimage.util import img_as_bool
from skimage.morphology import binary, footprints, gray
def test_binary_opening_anti_extensive():
    footprint = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
    result_default = binary.binary_opening(bw_img, footprint=footprint)
    assert np.all(result_default <= bw_img)
    result_max = binary.binary_opening(bw_img, footprint=footprint, mode='max')
    assert not np.all(result_max <= bw_img)