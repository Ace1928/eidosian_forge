from math import pi
import math
import cupy
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._iir_filter_conversions import (
def iirnotch(w0, Q, fs=2.0):
    """
    Design second-order IIR notch digital filter.

    A notch filter is a band-stop filter with a narrow bandwidth
    (high quality factor). It rejects a narrow frequency band and
    leaves the rest of the spectrum little changed.

    Parameters
    ----------
    w0 : float
        Frequency to remove from a signal. If `fs` is specified, this is in
        the same units as `fs`. By default, it is a normalized scalar that must
        satisfy  ``0 < w0 < 1``, with ``w0 = 1`` corresponding to half of the
        sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    See Also
    --------
    scipy.signal.iirnotch

    References
    ----------
    Sophocles J. Orfanidis, "Introduction To Signal Processing",
         Prentice-Hall, 1996
    """
    return _design_notch_peak_filter(w0, Q, 'notch', fs)