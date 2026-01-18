import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
@pytest.mark.parametrize('dtype', types)
def test_binary_dilation10(self, dtype):
    data = numpy.zeros([5], dtype)
    data[1] = 1
    out = ndimage.binary_dilation(data, origin=-1)
    assert_array_almost_equal(out, [0, 1, 1, 1, 0])