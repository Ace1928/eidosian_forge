import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
@pytest.mark.parametrize('dtype', types)
def test_binary_erosion07(self, dtype):
    data = numpy.ones([5], dtype)
    out = ndimage.binary_erosion(data)
    assert_array_almost_equal(out, [0, 1, 1, 1, 0])