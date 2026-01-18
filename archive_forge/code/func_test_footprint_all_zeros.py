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
def test_footprint_all_zeros():
    arr = numpy.random.randint(0, 100, (100, 100))
    kernel = numpy.zeros((3, 3), bool)
    with assert_raises(ValueError):
        ndimage.maximum_filter(arr, footprint=kernel)