import pytest
from numpy.testing import assert_allclose
from scipy.fft import fft
from scipy.signal import (czt, zoom_fft, czt_points, CZT, ZoomFFT)
import numpy as np
def check_czt(x):
    y = fft(x)
    y1 = czt(x)
    assert_allclose(y1, y, rtol=1e-13)
    y = fft(x, 100 * len(x))
    y1 = czt(x, 100 * len(x))
    assert_allclose(y1, y, rtol=1e-12)