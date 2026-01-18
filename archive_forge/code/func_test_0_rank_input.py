import pytest
from numpy.testing import assert_allclose
from scipy.fft import fft
from scipy.signal import (czt, zoom_fft, czt_points, CZT, ZoomFFT)
import numpy as np
def test_0_rank_input():
    with pytest.raises(IndexError, match='tuple index out of range'):
        czt(5)
    with pytest.raises(IndexError, match='tuple index out of range'):
        zoom_fft(5, 0.5)