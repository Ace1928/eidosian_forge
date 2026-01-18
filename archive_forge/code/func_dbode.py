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
def dbode(system, w=None, n=100):
    """
    Calculate Bode magnitude and phase data of a discrete-time system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `dlti`)
            * 2 (num, den, dt)
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

    Returns
    -------
    w : 1D ndarray
        Frequency array [rad/time_unit]
    mag : 1D ndarray
        Magnitude array [dB]
    phase : 1D ndarray
        Phase array [deg]

    See Also
    --------
    scipy.signal.dbode

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``z^2 + 3z + 5`` would be represented as ``[1, 3, 5]``).
    """
    w, y = dfreqresp(system, w=w, n=n)
    if isinstance(system, dlti):
        dt = system.dt
    else:
        dt = system[-1]
    mag = 20.0 * cupy.log10(abs(y))
    phase = cupy.rad2deg(cupy.unwrap(cupy.angle(y)))
    return (w / dt, mag, phase)