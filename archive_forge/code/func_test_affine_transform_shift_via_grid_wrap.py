import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('shift', [(1, 0), (0, 1), (-1, 1), (3, -5), (2, 7)])
@pytest.mark.parametrize('order', range(0, 6))
def test_affine_transform_shift_via_grid_wrap(self, shift, order):
    x = numpy.array([[0, 1], [2, 3]])
    affine = numpy.zeros((2, 3))
    affine[:2, :2] = numpy.eye(2)
    affine[:, 2] = shift
    assert_array_almost_equal(ndimage.affine_transform(x, affine, mode='grid-wrap', order=order), numpy.roll(x, shift, axis=(0, 1)))