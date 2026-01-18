import pytest
from numpy.testing import assert_allclose
from scipy.fft import fft
from scipy.signal import (czt, zoom_fft, czt_points, CZT, ZoomFFT)
import numpy as np
@pytest.mark.parametrize('m', [0, -11, 5.5, 4.0])
def test_czt_points_errors(m):
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        czt_points(m)