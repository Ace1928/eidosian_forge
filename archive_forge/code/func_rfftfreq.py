import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
def rfftfreq(n, d=1.0):
    """Return the FFT sample frequencies for real input.

    Args:
        n (int): Window length.
        d (scalar): Sample spacing.

    Returns:
        cupy.ndarray:
            Array of length ``n//2+1`` containing the sample frequencies.

    .. seealso:: :func:`numpy.fft.rfftfreq`
    """
    return cupy.arange(0, n // 2 + 1, dtype=np.float64) / (n * d)