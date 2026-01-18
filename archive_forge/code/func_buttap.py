import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def buttap(N):
    """Return (z,p,k) for analog prototype of Nth-order Butterworth filter.

    The filter will have an angular (e.g., rad/s) cutoff frequency of 1.

    See Also
    --------
    butter : Filter design function using this prototype
    scipy.signal.buttap

    """
    if abs(int(N)) != N:
        raise ValueError('Filter order must be a nonnegative integer')
    z = cupy.array([])
    m = cupy.arange(-N + 1, N, 2)
    p = -cupy.exp(1j * pi * m / (2 * N))
    k = 1
    return (z, p, k)