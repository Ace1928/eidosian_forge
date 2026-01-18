from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
def test_dct3_definition_ortho(mdata_x, rdt):
    x = np.array(mdata_x, dtype=rdt)
    dt = np.result_type(np.float32, rdt)
    y = dct(x, norm='ortho', type=2)
    xi = dct(y, norm='ortho', type=3)
    dec = dec_map[dct, rdt, 3]
    assert_equal(xi.dtype, dt)
    assert_array_almost_equal(xi, x, decimal=dec)