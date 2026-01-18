import math
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data
from skimage._shared.utils import _supported_float_type
from skimage.transform import pyramids
def test_check_factor():
    with pytest.raises(ValueError):
        pyramids._check_factor(0.99)
    with pytest.raises(ValueError):
        pyramids._check_factor(-2)