from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
def test_idst_definition(fftwdata_size, rdt, type, reference_data):
    xr, yr, dt = fftw_dst_ref(type, fftwdata_size, rdt, reference_data)
    x = idst(yr, type=type)
    dec = dec_map[idst, rdt, type]
    assert_equal(x.dtype, dt)
    assert_allclose(x, xr, rtol=0.0, atol=np.max(xr) * 10 ** (-dec))