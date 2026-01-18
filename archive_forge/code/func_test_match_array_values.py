import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from skimage import data
from skimage import exposure
from skimage._shared.utils import _supported_float_type
from skimage.exposure import histogram_matching
@pytest.mark.parametrize('array, template, expected_array', [(np.arange(10), np.arange(100), np.arange(9, 100, 10)), (np.random.rand(4), np.ones(3), np.ones(4))])
def test_match_array_values(array, template, expected_array):
    matched = histogram_matching._match_cumulative_cdf(array, template)
    assert_array_almost_equal(matched, expected_array)