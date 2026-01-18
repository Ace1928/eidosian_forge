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
def dlsim(system, u, t=None, x0=None):
    """
    Simulate output of a discrete-time linear system.

    Parameters
    ----------
    system : tuple of array_like or instance of `dlti`
        A tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1: (instance of `dlti`)
            * 3: (num, den, dt)
            * 4: (zeros, poles, gain, dt)
            * 5: (A, B, C, D, dt)

    u : array_like
        An input array describing the input at each time `t` (interpolation is
        assumed between given times).  If there are multiple inputs, then each
        column of the rank-2 array represents an input.
    t : array_like, optional
        The time steps at which the input is defined.  If `t` is given, it
        must be the same length as `u`, and the final value in `t` determines
        the number of steps returned in the output.
    x0 : array_like, optional
        The initial conditions on the state vector (zero by default).

    Returns
    -------
    tout : ndarray
        Time values for the output, as a 1-D array.
    yout : ndarray
        System response, as a 1-D array.
    xout : ndarray, optional
        Time-evolution of the state-vector.  Only generated if the input is a
        `StateSpace` system.

    See Also
    --------
    scipy.signal.dlsim
    lsim, dstep, dimpulse, cont2discrete
    """
    if isinstance(system, lti):
        raise AttributeError('dlsim can only be used with discrete-time dlti systems.')
    elif not isinstance(system, dlti):
        system = dlti(*system[:-1], dt=system[-1])
    is_ss_input = isinstance(system, StateSpace)
    system = system._as_ss()
    u = cupy.atleast_1d(u)
    if u.ndim == 1:
        u = cupy.atleast_2d(u).T
    if t is None:
        out_samples = len(u)
        stoptime = (out_samples - 1) * system.dt
    else:
        stoptime = t[-1]
        out_samples = int(cupy.floor(stoptime / system.dt)) + 1
    xout = cupy.zeros((out_samples, system.A.shape[0]))
    yout = cupy.zeros((out_samples, system.C.shape[0]))
    tout = cupy.linspace(0.0, stoptime, num=out_samples)
    if x0 is None:
        xout[0, :] = cupy.zeros((system.A.shape[1],))
    else:
        xout[0, :] = cupy.asarray(x0)
    if t is None:
        u_dt = u
    else:
        if len(u.shape) == 1:
            u = u[:, None]
        u_dt = make_interp_spline(t, u, k=1)(tout)
    for i in range(0, out_samples - 1):
        xout[i + 1, :] = system.A @ xout[i, :] + system.B @ u_dt[i, :]
        yout[i, :] = system.C @ xout[i, :] + system.D @ u_dt[i, :]
    yout[out_samples - 1, :] = system.C @ xout[out_samples - 1, :] + system.D @ u_dt[out_samples - 1, :]
    if is_ss_input:
        return (tout, yout, xout)
    else:
        return (tout, yout)