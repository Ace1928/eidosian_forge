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
def test_multiple_modes_laplace():
    arr = numpy.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    expected = numpy.array([[-2.0, 2.0, 1.0], [-2.0, -3.0, 2.0], [1.0, 1.0, 0.0]])
    modes = ['reflect', 'wrap']
    assert_equal(expected, ndimage.laplace(arr, mode=modes))