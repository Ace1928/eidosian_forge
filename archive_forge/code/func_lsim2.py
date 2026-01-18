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
def lsim2(system, U=None, T=None, X0=None, **kwargs):
    """
    Simulate output of a continuous-time linear system, by using
    the ODE solver `scipy.integrate.odeint`.

    .. deprecated:: 1.11.0
        Function `lsim2` is deprecated in favor of the faster `lsim` function.
        `lsim2` will be removed in SciPy 1.13.

    Parameters
    ----------
    system : an instance of the `lti` class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

        * 1: (instance of `lti`)
        * 2: (num, den)
        * 3: (zeros, poles, gain)
        * 4: (A, B, C, D)

    U : array_like (1D or 2D), optional
        An input array describing the input at each time T.  Linear
        interpolation is used between given times.  If there are
        multiple inputs, then each column of the rank-2 array
        represents an input.  If U is not given, the input is assumed
        to be zero.
    T : array_like (1D or 2D), optional
        The time steps at which the input is defined and at which the
        output is desired.  The default is 101 evenly spaced points on
        the interval [0,10.0].
    X0 : array_like (1D), optional
        The initial condition of the state vector.  If `X0` is not
        given, the initial conditions are assumed to be 0.
    kwargs : dict
        Additional keyword arguments are passed on to the function
        `odeint`.  See the notes below for more details.

    Returns
    -------
    T : 1D ndarray
        The time values for the output.
    yout : ndarray
        The response of the system.
    xout : ndarray
        The time-evolution of the state-vector.

    See Also
    --------
    lsim

    Notes
    -----
    This function uses `scipy.integrate.odeint` to solve the system's
    differential equations.  Additional keyword arguments given to `lsim2`
    are passed on to `scipy.integrate.odeint`.  See the documentation
    for `scipy.integrate.odeint` for the full list of arguments.

    As `lsim2` is now deprecated, users are advised to switch to the faster
    and more accurate `lsim` function. Keyword arguments for
    `scipy.integrate.odeint` are not supported in `lsim`, but not needed in
    general.

    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    Examples
    --------
    We'll use `lsim2` to simulate an analog Bessel filter applied to
    a signal.

    >>> import numpy as np
    >>> from scipy.signal import bessel, lsim2
    >>> import matplotlib.pyplot as plt

    Create a low-pass Bessel filter with a cutoff of 12 Hz.

    >>> b, a = bessel(N=5, Wn=2*np.pi*12, btype='lowpass', analog=True)

    Generate data to which the filter is applied.

    >>> t = np.linspace(0, 1.25, 500, endpoint=False)

    The input signal is the sum of three sinusoidal curves, with
    frequencies 4 Hz, 40 Hz, and 80 Hz.  The filter should mostly
    eliminate the 40 Hz and 80 Hz components, leaving just the 4 Hz signal.

    >>> u = (np.cos(2*np.pi*4*t) + 0.6*np.sin(2*np.pi*40*t) +
    ...      0.5*np.cos(2*np.pi*80*t))

    Simulate the filter with `lsim2`.

    >>> tout, yout, xout = lsim2((b, a), U=u, T=t)

    Plot the result.

    >>> plt.plot(t, u, 'r', alpha=0.5, linewidth=1, label='input')
    >>> plt.plot(tout, yout, 'k', linewidth=1.5, label='output')
    >>> plt.legend(loc='best', shadow=True, framealpha=1)
    >>> plt.grid(alpha=0.3)
    >>> plt.xlabel('t')
    >>> plt.show()

    In a second example, we simulate a double integrator ``y'' = u``, with
    a constant input ``u = 1``.  We'll use the state space representation
    of the integrator.

    >>> from scipy.signal import lti
    >>> A = np.array([[0, 1], [0, 0]])
    >>> B = np.array([[0], [1]])
    >>> C = np.array([[1, 0]])
    >>> D = 0
    >>> system = lti(A, B, C, D)

    `t` and `u` define the time and input signal for the system to
    be simulated.

    >>> t = np.linspace(0, 5, num=50)
    >>> u = np.ones_like(t)

    Compute the simulation, and then plot `y`.  As expected, the plot shows
    the curve ``y = 0.5*t**2``.

    >>> tout, y, x = lsim2(system, u, t)
    >>> plt.plot(t, y)
    >>> plt.grid(alpha=0.3)
    >>> plt.xlabel('t')
    >>> plt.show()

    """
    warnings.warn('lsim2 is deprecated and will be removed from scipy 1.13. Use the feature-equivalent lsim function.', DeprecationWarning, stacklevel=2)
    if isinstance(system, lti):
        sys = system._as_ss()
    elif isinstance(system, dlti):
        raise AttributeError('lsim2 can only be used with continuous-time systems.')
    else:
        sys = lti(*system)._as_ss()
    if X0 is None:
        X0 = zeros(sys.B.shape[0], sys.A.dtype)
    if T is None:
        T = linspace(0, 10.0, 101)
    T = atleast_1d(T)
    if len(T.shape) != 1:
        raise ValueError('T must be a rank-1 array.')
    if U is not None:
        U = atleast_1d(U)
        if len(U.shape) == 1:
            U = U.reshape(-1, 1)
        sU = U.shape
        if sU[0] != len(T):
            raise ValueError('U must have the same number of rows as elements in T.')
        if sU[1] != sys.inputs:
            raise ValueError('The number of inputs in U (%d) is not compatible with the number of system inputs (%d)' % (sU[1], sys.inputs))
        ufunc = interpolate.interp1d(T, U, kind='linear', axis=0, bounds_error=False)

        def fprime(x, t, sys, ufunc):
            """The vector field of the linear system."""
            return dot(sys.A, x) + squeeze(dot(sys.B, nan_to_num(ufunc(t))))
        xout = integrate.odeint(fprime, X0, T, args=(sys, ufunc), **kwargs)
        yout = dot(sys.C, transpose(xout)) + dot(sys.D, transpose(U))
    else:

        def fprime(x, t, sys):
            """The vector field of the linear system."""
            return dot(sys.A, x)
        xout = integrate.odeint(fprime, X0, T, args=(sys,), **kwargs)
        yout = dot(sys.C, transpose(xout))
    return (T, squeeze(transpose(yout)), xout)