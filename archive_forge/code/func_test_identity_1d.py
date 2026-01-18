import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.fft import dct, idct, dctn, idctn, dst, idst, dstn, idstn
import scipy.fft as fft
from scipy import fftpack
from scipy.conftest import (
from scipy._lib._array_api import copy, xp_assert_close
import math
@skip_if_array_api_gpu
@array_api_compatible
@pytest.mark.parametrize('forward, backward', [(dct, idct), (dst, idst)])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
@pytest.mark.parametrize('n', [2, 3, 4, 5, 10, 16])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('norm', [None, 'backward', 'ortho', 'forward'])
@pytest.mark.parametrize('orthogonalize', [False, True])
def test_identity_1d(forward, backward, type, n, axis, norm, orthogonalize, xp):
    x = xp.asarray(np.random.rand(n, n))
    y = forward(x, type, axis=axis, norm=norm, orthogonalize=orthogonalize)
    z = backward(y, type, axis=axis, norm=norm, orthogonalize=orthogonalize)
    xp_assert_close(z, x)
    pad = [(0, 0)] * 2
    pad[axis] = (0, 4)
    y2 = xp.asarray(np.pad(np.asarray(y), pad, mode='edge'))
    z2 = backward(y2, type, n, axis, norm, orthogonalize=orthogonalize)
    xp_assert_close(z2, x)