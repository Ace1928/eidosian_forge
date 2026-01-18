import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage._shared.utils import _supported_float_type
from skimage._shared.testing import assert_stacklevel
from skimage.filters import difference_of_gaussians, gaussian
def test_1d_ok():
    """Testing Gaussian Filter for 1D array.
    With any array consisting of positive integers and only one zero - it
    should filter all values to be greater than 0.1
    """
    nums = np.arange(7)
    filtered = gaussian(nums, preserve_range=True)
    assert np.all(filtered > 0.1)