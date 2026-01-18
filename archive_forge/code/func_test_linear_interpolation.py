import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
@pytest.mark.parametrize(['input_dtype', 'expected_dtype'], H_F_TYPE_CODES)
@pytest.mark.parametrize(['method', 'expected'], [('inverted_cdf', 20), ('averaged_inverted_cdf', 27.5), ('closest_observation', 20), ('interpolated_inverted_cdf', 20), ('hazen', 27.5), ('weibull', 26), ('linear', 29), ('median_unbiased', 27), ('normal_unbiased', 27.125)])
def test_linear_interpolation(self, method, expected, input_dtype, expected_dtype):
    expected_dtype = np.dtype(expected_dtype)
    if np._get_promotion_state() == 'legacy':
        expected_dtype = np.promote_types(expected_dtype, np.float64)
    arr = np.asarray([15.0, 20.0, 35.0, 40.0, 50.0], dtype=input_dtype)
    actual = np.percentile(arr, 40.0, method=method)
    np.testing.assert_almost_equal(actual, expected_dtype.type(expected), 14)
    if method in ['inverted_cdf', 'closest_observation']:
        if input_dtype == 'O':
            np.testing.assert_equal(np.asarray(actual).dtype, np.float64)
        else:
            np.testing.assert_equal(np.asarray(actual).dtype, np.dtype(input_dtype))
    else:
        np.testing.assert_equal(np.asarray(actual).dtype, np.dtype(expected_dtype))