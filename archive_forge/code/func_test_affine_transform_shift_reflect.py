import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
def test_affine_transform_shift_reflect(self, order):
    x = numpy.array([[0, 1, 2], [3, 4, 5]])
    affine = numpy.zeros((2, 3))
    affine[:2, :2] = numpy.eye(2)
    affine[:, 2] = x.shape
    assert_array_almost_equal(ndimage.affine_transform(x, affine, mode='reflect', order=order), x[::-1, ::-1])