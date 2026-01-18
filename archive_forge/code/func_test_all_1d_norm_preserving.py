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
@array_api_compatible
def test_all_1d_norm_preserving(self, xp):
    x = xp.asarray(random(30))
    xp_test = array_namespace(x)
    x_norm = xp_test.linalg.vector_norm(x)
    n = size(x) * 2
    func_pairs = [(fft.fft, fft.ifft), (fft.rfft, fft.irfft), (fft.ihfft, fft.hfft)]
    for forw, back in func_pairs:
        for n in [size(x), 2 * size(x)]:
            for norm in ['backward', 'ortho', 'forward']:
                tmp = forw(x, n=n, norm=norm)
                tmp = back(tmp, n=n, norm=norm)
                xp_assert_close(xp_test.linalg.vector_norm(tmp), x_norm)