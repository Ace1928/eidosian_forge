from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close, SCIPY_DEVICE
from scipy import fft
def test_next_fast_len(self):
    np.random.seed(1234)

    def nums():
        yield from range(1, 1000)
        yield (2 ** 5 * 3 ** 5 * 4 ** 5 + 1)
    for n in nums():
        m = next_fast_len(n)
        _assert_n_smooth(m, 11)
        assert m == next_fast_len(n, False)
        m = next_fast_len(n, True)
        _assert_n_smooth(m, 5)