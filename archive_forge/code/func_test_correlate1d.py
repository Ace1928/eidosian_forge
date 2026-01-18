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
def test_correlate1d(self):
    d = numpy.random.randn(5000)
    os = numpy.empty((4, d.size))
    ot = numpy.empty_like(os)
    k = numpy.arange(5)
    self.check_func_serial(4, ndimage.correlate1d, (d, k), os)
    self.check_func_thread(4, ndimage.correlate1d, (d, k), ot)
    assert_array_equal(os, ot)