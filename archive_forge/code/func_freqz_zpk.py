import operator
from math import pi
import warnings
import cupy
from cupy.polynomial.polynomial import (
import cupyx.scipy.fft as sp_fft
from cupyx import jit
from cupyx.scipy._lib._util import float_factorial
from cupyx.scipy.signal._polyutils import roots
def freqz_zpk(z, p, k, worN=512, whole=False, fs=2 * pi):
    """
    Compute the frequency response of a digital filter in ZPK form.

    Given the Zeros, Poles and Gain of a digital filter, compute its frequency
    response:

    :math:`H(z)=k \\prod_i (z - Z[i]) / \\prod_j (z - P[j])`

    where :math:`k` is the `gain`, :math:`Z` are the `zeros` and :math:`P` are
    the `poles`.

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512).

        If an array_like, compute the response at the frequencies given.
        These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs. Ignored if w is array_like.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    freqs : Compute the frequency response of an analog filter in TF form
    freqs_zpk : Compute the frequency response of an analog filter in ZPK form
    freqz : Compute the frequency response of a digital filter in TF form
    scipy.signal.freqz_zpk

    """
    z, p = map(cupy.atleast_1d, (z, p))
    if whole:
        lastpoint = 2 * pi
    else:
        lastpoint = pi
    if worN is None:
        w = cupy.linspace(0, lastpoint, 512, endpoint=False)
    else:
        N, _is_int = _try_convert_to_int(worN)
        if _is_int:
            w = cupy.linspace(0, lastpoint, N, endpoint=False)
        else:
            w = cupy.atleast_1d(worN)
            w = 2 * pi * w / fs
    zm1 = cupy.exp(1j * w)
    h = k * npp_polyvalfromroots(zm1, z) / npp_polyvalfromroots(zm1, p)
    w = w * fs / (2 * pi)
    return (w, h)