import operator
from math import pi
import warnings
import cupy
from cupy.polynomial.polynomial import (
import cupyx.scipy.fft as sp_fft
from cupyx import jit
from cupyx.scipy._lib._util import float_factorial
from cupyx.scipy.signal._polyutils import roots
def freqs_zpk(z, p, k, worN=200):
    """
    Compute frequency response of analog filter.

    Given the zeros `z`, poles `p`, and gain `k` of a filter, compute its
    frequency response::

                (jw-z[0]) * (jw-z[1]) * ... * (jw-z[-1])
     H(w) = k * ----------------------------------------
                (jw-p[0]) * (jw-p[1]) * ... * (jw-p[-1])

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    worN : {None, int, array_like}, optional
        If None, then compute at 200 frequencies around the interesting parts
        of the response curve (determined by pole-zero locations). If a single
        integer, then compute at that many frequencies. Otherwise, compute the
        response at the angular frequencies (e.g., rad/s) given in `worN`.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `h` was computed.
    h : ndarray
        The frequency response.

    See Also
    --------
    scipy.signal.freqs_zpk

    """
    k = cupy.asarray(k)
    if k.size > 1:
        raise ValueError('k must be a single scalar gain')
    if worN is None:
        w = findfreqs(z, p, 200, kind='zp')
    else:
        N, _is_int = _try_convert_to_int(worN)
        if _is_int:
            w = findfreqs(z, p, worN, kind='zp')
        else:
            w = worN
    w = cupy.atleast_1d(w)
    s = 1j * w
    num = npp_polyvalfromroots(s, z)
    den = npp_polyvalfromroots(s, p)
    h = k * num / den
    return (w, h)