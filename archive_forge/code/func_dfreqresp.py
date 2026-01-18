import warnings
import copy
from math import sqrt
import cupy
from cupyx.scipy import linalg
from cupyx.scipy.interpolate import make_interp_spline
from cupyx.scipy.linalg import expm, block_diag
from cupyx.scipy.signal._lti_conversion import (
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._filter_design import (
def dfreqresp(system, w=None, n=10000, whole=False):
    """
    Calculate the frequency response of a discrete-time system.

    Parameters
    ----------
    system : an instance of the `dlti` class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `dlti`)
            * 2 (numerator, denominator, dt)
            * 3 (zeros, poles, gain, dt)
            * 4 (A, B, C, D, dt)

    w : array_like, optional
        Array of frequencies (in radians/sample). Magnitude and phase data is
        calculated for every value in this array. If not given a reasonable
        set will be calculated.
    n : int, optional
        Number of frequency points to compute if `w` is not given. The `n`
        frequencies are logarithmically spaced in an interval chosen to
        include the influence of the poles and zeros of the system.
    whole : bool, optional
        Normally, if 'w' is not given, frequencies are computed from 0 to the
        Nyquist frequency, pi radians/sample (upper-half of unit-circle). If
        `whole` is True, compute frequencies from 0 to 2*pi radians/sample.

    Returns
    -------
    w : 1D ndarray
        Frequency array [radians/sample]
    H : 1D ndarray
        Array of complex magnitude values

    See Also
    --------
    scipy.signal.dfeqresp

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``z^2 + 3z + 5`` would be represented as ``[1, 3, 5]``).
    """
    if not isinstance(system, dlti):
        if isinstance(system, lti):
            raise AttributeError('dfreqresp can only be used with discrete-time systems.')
        system = dlti(*system[:-1], dt=system[-1])
    if isinstance(system, StateSpace):
        system = system._as_tf()
    if not isinstance(system, (TransferFunction, ZerosPolesGain)):
        raise ValueError('Unknown system type')
    if system.inputs != 1 or system.outputs != 1:
        raise ValueError('dfreqresp requires a SISO (single input, single output) system.')
    if w is not None:
        worN = w
    else:
        worN = n
    if isinstance(system, TransferFunction):
        num, den = TransferFunction._z_to_zinv(system.num.ravel(), system.den)
        w, h = freqz(num, den, worN=worN, whole=whole)
    elif isinstance(system, ZerosPolesGain):
        w, h = freqz_zpk(system.zeros, system.poles, system.gain, worN=worN, whole=whole)
    return (w, h)