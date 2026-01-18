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
@pytest.mark.parametrize('x, axis, expected_avg, weights, expected_wavg, expected_wsum', [([1, 2, 3], None, [2.0], [3, 4, 1], [1.75], [8.0]), ([[1, 2, 5], [1, 6, 11]], 0, [[1.0, 4.0, 8.0]], [1, 3], [[1.0, 5.0, 9.5]], [[4, 4, 4]])])
def test_basic_keepdims(self, x, axis, expected_avg, weights, expected_wavg, expected_wsum):
    avg = np.average(x, axis=axis, keepdims=True)
    assert avg.shape == np.shape(expected_avg)
    assert_array_equal(avg, expected_avg)
    wavg = np.average(x, axis=axis, weights=weights, keepdims=True)
    assert wavg.shape == np.shape(expected_wavg)
    assert_array_equal(wavg, expected_wavg)
    wavg, wsum = np.average(x, axis=axis, weights=weights, returned=True, keepdims=True)
    assert wavg.shape == np.shape(expected_wavg)
    assert_array_equal(wavg, expected_wavg)
    assert wsum.shape == np.shape(expected_wsum)
    assert_array_equal(wsum, expected_wsum)