import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('mode, expected_value', [('nearest', [1, 1, 2, 3]), ('wrap', [3, 1, 2, 3]), ('grid-wrap', [4, 1, 2, 3]), ('mirror', [2, 1, 2, 3]), ('reflect', [1, 1, 2, 3]), ('constant', [-1, 1, 2, 3]), ('grid-constant', [-1, 1, 2, 3])])
def test_boundaries2(self, mode, expected_value):

    def shift(x):
        return (x[0] - 0.9,)
    data = numpy.array([1, 2, 3, 4])
    assert_array_equal(expected_value, ndimage.geometric_transform(data, shift, cval=-1, mode=mode, output_shape=(4,)))