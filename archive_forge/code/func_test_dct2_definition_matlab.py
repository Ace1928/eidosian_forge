from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
def test_dct2_definition_matlab(mdata_xy, rdt):
    dt = np.result_type(np.float32, rdt)
    x = np.array(mdata_xy[0], dtype=dt)
    yr = mdata_xy[1]
    y = dct(x, norm='ortho', type=2)
    dec = dec_map[dct, rdt, 2]
    assert_equal(y.dtype, dt)
    assert_array_almost_equal(y, yr, decimal=dec)