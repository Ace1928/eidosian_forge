import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
@pytest.mark.parametrize('dtype', types)
def test_binary_dilation06(self, dtype):
    data = numpy.zeros([3], dtype)
    out = ndimage.binary_dilation(data)
    assert_array_almost_equal(out, [0, 0, 0])