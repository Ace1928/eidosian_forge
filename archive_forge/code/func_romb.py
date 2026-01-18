from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, cast
import numpy as np
import numpy.typing as npt
import math
import warnings
from collections import namedtuple
from scipy.special import roots_legendre
from scipy.special import gammaln, logsumexp
from scipy._lib._util import _rng_spawn
from scipy._lib.deprecation import (_NoValue, _deprecate_positional_args,
def romb(y, dx=1.0, axis=-1, show=False):
    """
    Romberg integration using samples of a function.

    Parameters
    ----------
    y : array_like
        A vector of ``2**k + 1`` equally-spaced samples of a function.
    dx : float, optional
        The sample spacing. Default is 1.
    axis : int, optional
        The axis along which to integrate. Default is -1 (last axis).
    show : bool, optional
        When `y` is a single 1-D array, then if this argument is True
        print the table showing Richardson extrapolation from the
        samples. Default is False.

    Returns
    -------
    romb : ndarray
        The integrated result for `axis`.

    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    romberg : adaptive Romberg quadrature
    quadrature : adaptive Gaussian quadrature
    fixed_quad : fixed-order Gaussian quadrature
    dblquad : double integrals
    tplquad : triple integrals
    simpson : integrators for sampled data
    cumulative_trapezoid : cumulative integration for sampled data
    ode : ODE integrators
    odeint : ODE integrators

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> x = np.arange(10, 14.25, 0.25)
    >>> y = np.arange(3, 12)

    >>> integrate.romb(y)
    56.0

    >>> y = np.sin(np.power(x, 2.5))
    >>> integrate.romb(y)
    -0.742561336672229

    >>> integrate.romb(y, show=True)
    Richardson Extrapolation Table for Romberg Integration
    ======================================================
    -0.81576
     4.63862  6.45674
    -1.10581 -3.02062 -3.65245
    -2.57379 -3.06311 -3.06595 -3.05664
    -1.34093 -0.92997 -0.78776 -0.75160 -0.74256
    ======================================================
    -0.742561336672229  # may vary

    """
    y = np.asarray(y)
    nd = len(y.shape)
    Nsamps = y.shape[axis]
    Ninterv = Nsamps - 1
    n = 1
    k = 0
    while n < Ninterv:
        n <<= 1
        k += 1
    if n != Ninterv:
        raise ValueError('Number of samples must be one plus a non-negative power of 2.')
    R = {}
    slice_all = (slice(None),) * nd
    slice0 = tupleset(slice_all, axis, 0)
    slicem1 = tupleset(slice_all, axis, -1)
    h = Ninterv * np.asarray(dx, dtype=float)
    R[0, 0] = (y[slice0] + y[slicem1]) / 2.0 * h
    slice_R = slice_all
    start = stop = step = Ninterv
    for i in range(1, k + 1):
        start >>= 1
        slice_R = tupleset(slice_R, axis, slice(start, stop, step))
        step >>= 1
        R[i, 0] = 0.5 * (R[i - 1, 0] + h * y[slice_R].sum(axis=axis))
        for j in range(1, i + 1):
            prev = R[i, j - 1]
            R[i, j] = prev + (prev - R[i - 1, j - 1]) / ((1 << 2 * j) - 1)
        h /= 2.0
    if show:
        if not np.isscalar(R[0, 0]):
            print('*** Printing table only supported for integrals' + ' of a single data set.')
        else:
            try:
                precis = show[0]
            except (TypeError, IndexError):
                precis = 5
            try:
                width = show[1]
            except (TypeError, IndexError):
                width = 8
            formstr = '%%%d.%df' % (width, precis)
            title = 'Richardson Extrapolation Table for Romberg Integration'
            print(title, '=' * len(title), sep='\n', end='\n')
            for i in range(k + 1):
                for j in range(i + 1):
                    print(formstr % R[i, j], end=' ')
                print()
            print('=' * len(title))
    return R[k, k]