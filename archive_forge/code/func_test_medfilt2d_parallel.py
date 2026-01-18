import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from itertools import product
from math import gcd
import pytest
from pytest import raises as assert_raises
from numpy.testing import (
from numpy import array, arange
import numpy as np
from scipy.fft import fft
from scipy.ndimage import correlate1d
from scipy.optimize import fmin, linear_sum_assignment
from scipy import signal
from scipy.signal import (
from scipy.signal.windows import hann
from scipy.signal._signaltools import (_filtfilt_gust, _compute_factors,
from scipy.signal._upfirdn import _upfirdn_modes
from scipy._lib import _testutils
from scipy._lib._util import ComplexWarning, np_long, np_ulong
@pytest.mark.parametrize('dtype', [np.ubyte, np.float32, np.float64])
def test_medfilt2d_parallel(self, dtype):
    in_typed = np.array(self.IN, dtype=dtype)
    expected = np.array(self.OUT, dtype=dtype)
    assert in_typed.shape == expected.shape
    M1 = expected.shape[0] // 2
    N1 = expected.shape[1] // 2
    offM = self.KERNEL_SIZE[0] // 2 + 1
    offN = self.KERNEL_SIZE[1] // 2 + 1

    def apply(chunk):
        M, N = chunk
        if M == 0:
            Min = slice(0, M1 + offM)
            Msel = slice(0, -offM)
            Mout = slice(0, M1)
        else:
            Min = slice(M1 - offM, None)
            Msel = slice(offM, None)
            Mout = slice(M1, None)
        if N == 0:
            Nin = slice(0, N1 + offN)
            Nsel = slice(0, -offN)
            Nout = slice(0, N1)
        else:
            Nin = slice(N1 - offN, None)
            Nsel = slice(offN, None)
            Nout = slice(N1, None)
        chunk_data = in_typed[Min, Nin]
        med = signal.medfilt2d(chunk_data, self.KERNEL_SIZE)
        return (med[Msel, Nsel], Mout, Nout)
    output = np.zeros_like(expected)
    with ThreadPoolExecutor(max_workers=4) as pool:
        chunks = {(0, 0), (0, 1), (1, 0), (1, 1)}
        futures = {pool.submit(apply, chunk) for chunk in chunks}
        for future in as_completed(futures):
            data, Mslice, Nslice = future.result()
            output[Mslice, Nslice] = data
    assert_array_equal(output, expected)