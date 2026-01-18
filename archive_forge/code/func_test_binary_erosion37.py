import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_binary_erosion37(self):
    a = numpy.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
    b = numpy.zeros_like(a)
    out = ndimage.binary_erosion(a, structure=a, output=b, iterations=0, border_value=True, brute_force=True)
    assert_(out is b)
    assert_array_equal(ndimage.binary_erosion(a, structure=a, iterations=0, border_value=True), b)