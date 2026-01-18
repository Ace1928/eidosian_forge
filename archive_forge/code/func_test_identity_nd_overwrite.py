import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.fft import dct, idct, dctn, idctn, dst, idst, dstn, idstn
import scipy.fft as fft
from scipy import fftpack
from scipy.conftest import (
from scipy._lib._array_api import copy, xp_assert_close
import math
@pytest.mark.parametrize('forward, backward', [(dctn, idctn), (dstn, idstn)])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
@pytest.mark.parametrize('shape, axes', [((4, 5), 0), ((4, 5), 1), ((4, 5), None)])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize('norm', [None, 'backward', 'ortho', 'forward'])
@pytest.mark.parametrize('overwrite_x', [False, True])
def test_identity_nd_overwrite(forward, backward, type, shape, axes, dtype, norm, overwrite_x):
    x = np.random.random(shape).astype(dtype)
    x_orig = x.copy()
    if axes is not None:
        shape = np.take(shape, axes)
    y = forward(x, type, axes=axes, norm=norm)
    y_orig = y.copy()
    z = backward(y, type, axes=axes, norm=norm)
    if overwrite_x:
        assert_allclose(z, x_orig, rtol=1e-06, atol=1e-06)
    else:
        assert_allclose(z, x, rtol=1e-06, atol=1e-06)
        assert_array_equal(x, x_orig)
        assert_array_equal(y, y_orig)