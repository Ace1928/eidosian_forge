import pytest
from numpy.testing import assert_allclose
from scipy.fft import fft
from scipy.signal import (czt, zoom_fft, czt_points, CZT, ZoomFFT)
import numpy as np
def test_czt_points():
    for N in (1, 2, 3, 8, 11, 100, 101, 10007):
        assert_allclose(czt_points(N), np.exp(2j * np.pi * np.arange(N) / N), rtol=1e-30)
    assert_allclose(czt_points(7, w=1), np.ones(7), rtol=1e-30)
    assert_allclose(czt_points(11, w=2.0), 1 / 2 ** np.arange(11), rtol=1e-30)
    func = CZT(12, m=11, w=2.0, a=1)
    assert_allclose(func.points(), 1 / 2 ** np.arange(11), rtol=1e-30)