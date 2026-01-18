import functools
import itertools
import math
import numpy
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d
from . import types, float_types, complex_types
@pytest.mark.parametrize('dtype', types)
def test_generic_filter01(self, dtype):
    filter_ = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    footprint = numpy.array([[1, 0], [0, 1]])
    cf = numpy.array([1.0, 4.0])

    def _filter_func(buffer, weights, total=1.0):
        weights = cf / total
        return (buffer * weights).sum()
    a = numpy.arange(12, dtype=dtype)
    a.shape = (3, 4)
    r1 = ndimage.correlate(a, filter_ * footprint)
    if dtype in float_types:
        r1 /= 5
    else:
        r1 //= 5
    r2 = ndimage.generic_filter(a, _filter_func, footprint=footprint, extra_arguments=(cf,), extra_keywords={'total': cf.sum()})
    assert_array_almost_equal(r1, r2)
    with assert_raises(RuntimeError):
        r2 = ndimage.generic_filter(a, _filter_func, mode=['reflect', 'reflect'], footprint=footprint, extra_arguments=(cf,), extra_keywords={'total': cf.sum()})