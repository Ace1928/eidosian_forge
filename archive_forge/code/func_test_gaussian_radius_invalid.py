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
def test_gaussian_radius_invalid():
    with assert_raises(ValueError):
        ndimage.gaussian_filter1d(numpy.zeros(8), sigma=1, radius=-1)
    with assert_raises(ValueError):
        ndimage.gaussian_filter1d(numpy.zeros(8), sigma=1, radius=1.1)