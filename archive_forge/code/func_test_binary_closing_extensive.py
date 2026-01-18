import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage import data, color, morphology
from skimage.util import img_as_bool
from skimage.morphology import binary, footprints, gray
def test_binary_closing_extensive():
    footprint = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
    result_default = binary.binary_closing(bw_img, footprint=footprint)
    assert np.all(result_default >= bw_img)
    result_min = binary.binary_closing(img, footprint=footprint, mode='min')
    assert not np.all(result_min >= bw_img)