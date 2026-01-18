import warnings
from scipy.linalg import qr as s_qr
from scipy import integrate, interpolate, linalg
from scipy.interpolate import make_interp_spline
from ._filter_design import (tf2zpk, zpk2tf, normalize, freqs, freqz, freqs_zpk,
from ._lti_conversion import (tf2ss, abcd_normalize, ss2tf, zpk2ss, ss2zpk,
import numpy
import numpy as np
from numpy import (real, atleast_1d, squeeze, asarray, zeros,
import copy
def step2(system, X0=None, T=None, N=None, **kwargs):
    """Step response of continuous-time system.

    This function is functionally the same as `scipy.signal.step`, but
    it uses the function `scipy.signal.lsim2` to compute the step
    response.

    .. deprecated:: 1.11.0
        Function `step2` is deprecated in favor of the faster `step` function.
        `step2` will be removed in SciPy 1.13.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple of array_like
        describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `lti`)
            * 2 (num, den)
            * 3 (zeros, poles, gain)
            * 4 (A, B, C, D)

    X0 : array_like, optional
        Initial state-vector (default is zero).
    T : array_like, optional
        Time points (computed if not given).
    N : int, optional
        Number of time points to compute if `T` is not given.
    kwargs : various types
        Additional keyword arguments are passed on the function
        `scipy.signal.lsim2`, which in turn passes them on to
        `scipy.integrate.odeint`.  See the documentation for
        `scipy.integrate.odeint` for information about these arguments.

    Returns
    -------
    T : 1D ndarray
        Output time points.
    yout : 1D ndarray
        Step response of system.

    See Also
    --------
    scipy.signal.step

    Notes
    -----
    As `step2` is now deprecated, users are advised to switch to the faster
    and more accurate `step` function. Keyword arguments for
    `scipy.integrate.odeint` are not supported in `step`, but not needed in
    general.

    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    .. versionadded:: 0.8.0

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> lti = signal.lti([1.0], [1.0, 1.0])
    >>> t, y = signal.step2(lti)

    >>> plt.plot(t, y)
    >>> plt.xlabel('Time [s]')
    >>> plt.ylabel('Amplitude')
    >>> plt.title('Step response for 1. Order Lowpass')
    >>> plt.grid()

    """
    warnings.warn('step2 is deprecated and will be removed from scipy 1.13. Use the feature-equivalent step function.', DeprecationWarning, stacklevel=2)
    if isinstance(system, lti):
        sys = system._as_ss()
    elif isinstance(system, dlti):
        raise AttributeError('step2 can only be used with continuous-time systems.')
    else:
        sys = lti(*system)._as_ss()
    if N is None:
        N = 100
    if T is None:
        T = _default_response_times(sys.A, N)
    else:
        T = asarray(T)
    U = ones(T.shape, sys.A.dtype)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'lsim2 is deprecated and will be removed from scipy 1.13', DeprecationWarning)
        vals = lsim2(sys, U, T, X0=X0, **kwargs)
    return (vals[0], vals[1])