import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
@pytest.mark.parametrize('dtype', types)
def test_binary_erosion10(self, dtype):
    data = numpy.ones([5], dtype)
    data[2] = 0
    out = ndimage.binary_erosion(data, border_value=1)
    assert_array_almost_equal(out, [1, 0, 0, 0, 1])