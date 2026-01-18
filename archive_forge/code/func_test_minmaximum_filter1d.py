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
def test_minmaximum_filter1d():
    in_ = numpy.arange(10)
    out = ndimage.minimum_filter1d(in_, 1)
    assert_equal(in_, out)
    out = ndimage.maximum_filter1d(in_, 1)
    assert_equal(in_, out)
    out = ndimage.minimum_filter1d(in_, 5, mode='reflect')
    assert_equal([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], out)
    out = ndimage.maximum_filter1d(in_, 5, mode='reflect')
    assert_equal([2, 3, 4, 5, 6, 7, 8, 9, 9, 9], out)
    out = ndimage.minimum_filter1d(in_, 5, mode='constant', cval=-1)
    assert_equal([-1, -1, 0, 1, 2, 3, 4, 5, -1, -1], out)
    out = ndimage.maximum_filter1d(in_, 5, mode='constant', cval=10)
    assert_equal([10, 10, 4, 5, 6, 7, 8, 9, 10, 10], out)
    out = ndimage.minimum_filter1d(in_, 5, mode='nearest')
    assert_equal([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], out)
    out = ndimage.maximum_filter1d(in_, 5, mode='nearest')
    assert_equal([2, 3, 4, 5, 6, 7, 8, 9, 9, 9], out)
    out = ndimage.minimum_filter1d(in_, 5, mode='wrap')
    assert_equal([0, 0, 0, 1, 2, 3, 4, 5, 0, 0], out)
    out = ndimage.maximum_filter1d(in_, 5, mode='wrap')
    assert_equal([9, 9, 4, 5, 6, 7, 8, 9, 9, 9], out)