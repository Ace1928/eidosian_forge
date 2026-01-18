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
def test_correlate26(self):
    y = ndimage.convolve1d(numpy.ones(1), numpy.ones(5), mode='mirror')
    assert_array_equal(y, numpy.array(5.0))
    y = ndimage.correlate1d(numpy.ones(1), numpy.ones(5), mode='mirror')
    assert_array_equal(y, numpy.array(5.0))