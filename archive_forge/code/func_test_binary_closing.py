import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage import data, color, morphology
from skimage.util import img_as_bool
from skimage.morphology import binary, footprints, gray
def test_binary_closing():
    footprint = morphology.square(3)
    binary_res = binary.binary_closing(bw_img, footprint)
    gray_res = img_as_bool(gray.closing(bw_img, footprint))
    assert_array_equal(binary_res, gray_res)