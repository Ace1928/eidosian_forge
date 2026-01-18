from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
@pytest.mark.parametrize('dtype', dtypes)
@pytest.mark.parametrize('overwrite_x', [True, False])
@pytest.mark.parametrize('shape,axes', [((16,), None), ((16,), (0,)), ((16, 2), (0,)), ((2, 16), (1,)), ((8, 16), None), ((8, 16), (0, 1)), ((8, 16, 2), (0, 1)), ((8, 16, 2), (1, 2)), ((8, 16, 2), (0,)), ((8, 16, 2), (1,)), ((8, 16, 2), (2,)), ((8, 16, 2), None), ((8, 16, 2), (0, 1, 2))])
def test_fftn_ifftn(self, dtype, overwrite_x, shape, axes):
    overwritable = (np.clongdouble, np.complex128, np.complex64)
    self._check_nd_one(fftn, dtype, shape, axes, overwritable, overwrite_x)
    self._check_nd_one(ifftn, dtype, shape, axes, overwritable, overwrite_x)