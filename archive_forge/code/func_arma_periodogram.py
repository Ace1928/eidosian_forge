from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
def arma_periodogram(ar, ma, worN=None, whole=0):
    """
    Periodogram for ARMA process given by lag-polynomials ar and ma.

    Parameters
    ----------
    ar : array_like
        The autoregressive lag-polynomial with leading 1 and lhs sign.
    ma : array_like
        The moving average lag-polynomial with leading 1.
    worN : {None, int}, optional
        An option for scipy.signal.freqz (read "w or N").
        If None, then compute at 512 frequencies around the unit circle.
        If a single integer, the compute at that many frequencies.
        Otherwise, compute the response at frequencies given in worN.
    whole : {0,1}, optional
        An options for scipy.signal.freqz/
        Normally, frequencies are computed from 0 to pi (upper-half of
        unit-circle.  If whole is non-zero compute frequencies from 0 to 2*pi.

    Returns
    -------
    w : ndarray
        The frequencies.
    sd : ndarray
        The periodogram, also known as the spectral density.

    Notes
    -----
    Normalization ?

    This uses signal.freqz, which does not use fft. There is a fft version
    somewhere.
    """
    w, h = signal.freqz(ma, ar, worN=worN, whole=whole)
    sd = np.abs(h) ** 2 / np.sqrt(2 * np.pi)
    if np.any(np.isnan(h)):
        import warnings
        warnings.warn('Warning: nan in frequency response h, maybe a unit root', RuntimeWarning, stacklevel=2)
    return (w, sd)