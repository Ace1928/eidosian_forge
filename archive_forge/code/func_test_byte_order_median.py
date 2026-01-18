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
def test_byte_order_median():
    """Regression test for #413: median_filter does not handle bytes orders."""
    a = numpy.arange(9, dtype='<f4').reshape(3, 3)
    ref = ndimage.median_filter(a, (3, 3))
    b = numpy.arange(9, dtype='>f4').reshape(3, 3)
    t = ndimage.median_filter(b, (3, 3))
    assert_array_almost_equal(ref, t)