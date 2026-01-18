import warnings
import numpy as np
import pytest
from numpy.testing import (
from packaging.version import Version
from skimage import data
from skimage import exposure
from skimage import util
from skimage.color import rgb2gray
from skimage.exposure.exposure import intensity_range
from skimage.util.dtype import dtype_range
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
def test_adjust_inv_sigmoid_cutoff_half():
    """Verifying the output with expected results for inverse sigmoid
    correction with cutoff equal to half and gain of 10"""
    image = np.arange(0, 255, 4, np.uint8).reshape((8, 8))
    expected = np.array([[253, 253, 252, 252, 251, 251, 250, 249], [249, 248, 247, 245, 244, 242, 240, 238], [235, 232, 229, 225, 220, 215, 210, 204], [197, 190, 182, 174, 165, 155, 146, 136], [126, 116, 106, 96, 87, 78, 70, 62], [55, 49, 43, 37, 33, 28, 25, 21], [18, 16, 14, 12, 10, 8, 7, 6], [5, 4, 4, 3, 3, 2, 2, 1]], dtype=np.uint8)
    result = exposure.adjust_sigmoid(image, 0.5, 10, True)
    assert_array_equal(result, expected)