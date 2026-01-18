import queue
import threading
import multiprocessing
import numpy as np
import pytest
from numpy.random import random
from numpy.testing import assert_array_almost_equal, assert_allclose
from pytest import raises as assert_raises
import scipy.fft as fft
from scipy.conftest import (
from scipy._lib._array_api import (
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble, np.complex64, np.complex128, np.clongdouble])
@pytest.mark.parametrize('order', ['F', 'non-contiguous'])
@pytest.mark.parametrize('fft', [fft.fft, fft.fft2, fft.fftn, fft.ifft, fft.ifft2, fft.ifftn])
def test_fft_with_order(dtype, order, fft):
    rng = np.random.RandomState(42)
    X = rng.rand(8, 7, 13).astype(dtype, copy=False)
    if order == 'F':
        Y = np.asfortranarray(X)
    else:
        Y = X[::-1]
        X = np.ascontiguousarray(X[::-1])
    if fft.__name__.endswith('fft'):
        for axis in range(3):
            X_res = fft(X, axis=axis)
            Y_res = fft(Y, axis=axis)
            assert_array_almost_equal(X_res, Y_res)
    elif fft.__name__.endswith(('fft2', 'fftn')):
        axes = [(0, 1), (1, 2), (0, 2)]
        if fft.__name__.endswith('fftn'):
            axes.extend([(0,), (1,), (2,), None])
        for ax in axes:
            X_res = fft(X, axes=ax)
            Y_res = fft(Y, axes=ax)
            assert_array_almost_equal(X_res, Y_res)
    else:
        raise ValueError