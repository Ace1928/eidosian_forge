import queue
import threading
import multiprocessing
import numpy as np
import pytest
from numpy.random import random
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.fft as fft
@pytest.mark.parametrize('op', [fft.fft2, fft.ifft2, fft.rfft2, fft.irfft2, fft.hfft2, fft.ihfft2, fft.fftn, fft.ifftn, fft.rfftn, fft.irfftn, fft.hfftn, fft.ihfftn])
def test_axes_subset_with_shape(self, op):
    x = random((16, 8, 4))
    axes = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    for a in axes:
        shape = tuple([2 * x.shape[ax] if ax in a[:2] else x.shape[ax] for ax in range(x.ndim)])
        op_tr = op(np.transpose(x, a), s=shape[:2], axes=(0, 1))
        tr_op = np.transpose(op(x, s=shape[:2], axes=a[:2]), a)
        assert_array_almost_equal(op_tr, tr_op)