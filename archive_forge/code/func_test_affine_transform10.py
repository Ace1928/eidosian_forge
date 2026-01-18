import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_affine_transform10(self, order):
    data = numpy.ones([2], numpy.float64)
    out = ndimage.affine_transform(data, [[0.5]], output_shape=(4,), order=order)
    assert_array_almost_equal(out, [1, 1, 1, 0])