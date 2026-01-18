import math
from itertools import product
from typing import cast, get_args, Literal
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.fft import fftshift
from scipy.stats import norm as normal_distribution  # type: ignore
from scipy.signal import get_window, welch, stft, istft, spectrogram
from scipy.signal._short_time_fft import FFT_MODE_TYPE, \
from scipy.signal.windows import gaussian
def test__calc_dual_canonical_window_exceptions():
    """Raise all exceptions in `_calc_dual_canonical_window`."""
    with pytest.raises(ValueError, match='hop=5 is larger than window len.*'):
        _calc_dual_canonical_window(np.ones(4), 5)
    with pytest.raises(ValueError, match='.* Transform not invertible!'):
        _calc_dual_canonical_window(np.array([0.1, 0.2, 0.3, 0]), 4)
    with pytest.raises(ValueError, match="Parameter 'win' cannot be of int.*"):
        _calc_dual_canonical_window(np.ones(4, dtype=int), 1)