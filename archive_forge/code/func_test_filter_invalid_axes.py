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
@pytest.mark.parametrize('filter_func, args', [(ndimage.gaussian_filter, (1.0,)), (ndimage.uniform_filter, (3,)), (ndimage.minimum_filter, (3,)), (ndimage.maximum_filter, (3,)), (ndimage.median_filter, (3,)), (ndimage.rank_filter, (2, 3)), (ndimage.percentile_filter, (30, 3))])
@pytest.mark.parametrize('axes', [(1.5,), (0, 1, 2, 3), (3,), (-4,)])
def test_filter_invalid_axes(self, filter_func, args, axes):
    array = numpy.arange(6 * 8 * 12, dtype=numpy.float64).reshape(6, 8, 12)
    if any((isinstance(ax, float) for ax in axes)):
        error_class = TypeError
        match = 'cannot be interpreted as an integer'
    else:
        error_class = ValueError
        match = 'out of range'
    with pytest.raises(error_class, match=match):
        filter_func(array, *args, axes=axes)