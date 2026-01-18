import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_affine_transform25(self, order):
    data = numpy.array([4, 1, 3, 2])
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'The behavior of affine_transform with a 1-D array .* has changed')
        out1 = ndimage.affine_transform(data, [0.5], -1, order=order)
    out2 = ndimage.affine_transform(data, [[0.5]], -1, order=order)
    assert_array_almost_equal(out1, out2)