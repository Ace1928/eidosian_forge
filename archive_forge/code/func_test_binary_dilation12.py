import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
@pytest.mark.parametrize('dtype', types)
def test_binary_dilation12(self, dtype):
    data = numpy.zeros([5], dtype)
    data[1] = 1
    struct = [1, 0, 1]
    out = ndimage.binary_dilation(data, struct)
    assert_array_almost_equal(out, [1, 0, 1, 0, 0])