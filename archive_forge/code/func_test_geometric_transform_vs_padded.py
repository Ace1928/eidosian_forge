import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('mode', ['grid-constant', 'grid-wrap', 'nearest', 'mirror', 'reflect'])
@pytest.mark.parametrize('order', range(6))
def test_geometric_transform_vs_padded(self, order, mode):
    x = numpy.arange(144, dtype=float).reshape(12, 12)

    def mapping(x):
        return (x[0] - 0.4, x[1] + 2.3)
    npad = 24
    pad_mode = ndimage_to_numpy_mode.get(mode)
    xp = numpy.pad(x, npad, mode=pad_mode)
    center_slice = tuple([slice(npad, -npad)] * x.ndim)
    expected_result = ndimage.geometric_transform(xp, mapping, mode=mode, order=order)[center_slice]
    assert_allclose(ndimage.geometric_transform(x, mapping, mode=mode, order=order), expected_result, rtol=1e-07)