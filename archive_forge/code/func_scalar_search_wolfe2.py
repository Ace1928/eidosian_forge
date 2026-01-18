from warnings import warn
from scipy.optimize import _minpack2 as minpack2    # noqa: F401
from ._dcsrch import DCSRCH
import numpy as np
def scalar_search_wolfe2(phi, derphi, phi0=None, old_phi0=None, derphi0=None, c1=0.0001, c2=0.9, amax=None, extra_condition=None, maxiter=10):
    """Find alpha that satisfies strong Wolfe conditions.

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable phi(alpha)
        Objective scalar function.
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0.
    old_phi0 : float, optional
        Value of phi at previous point.
    derphi0 : float, optional
        Value of derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size.
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, phi_value)``
        returning a boolean. The line search accepts the value
        of ``alpha`` only if this callable returns ``True``.
        If the callable returns ``False`` for the step length,
        the algorithm will continue with new iterates.
        The callable is only called for iterates satisfying
        the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    alpha_star : float or None
        Best alpha, or None if the line search algorithm did not converge.
    phi_star : float
        phi at alpha_star.
    phi0 : float
        phi at 0.
    derphi_star : float or None
        derphi at alpha_star, or None if the line search algorithm
        did not converge.

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions. See Wright and Nocedal, 'Numerical Optimization',
    1999, pp. 59-61.

    """
    _check_c1_c2(c1, c2)
    if phi0 is None:
        phi0 = phi(0.0)
    if derphi0 is None:
        derphi0 = derphi(0.0)
    alpha0 = 0
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01 * 2 * (phi0 - old_phi0) / derphi0)
    else:
        alpha1 = 1.0
    if alpha1 < 0:
        alpha1 = 1.0
    if amax is not None:
        alpha1 = min(alpha1, amax)
    phi_a1 = phi(alpha1)
    phi_a0 = phi0
    derphi_a0 = derphi0
    if extra_condition is None:

        def extra_condition(alpha, phi):
            return True
    for i in range(maxiter):
        if alpha1 == 0 or (amax is not None and alpha0 == amax):
            alpha_star = None
            phi_star = phi0
            phi0 = old_phi0
            derphi_star = None
            if alpha1 == 0:
                msg = 'Rounding errors prevent the line search from converging'
            else:
                msg = 'The line search algorithm could not find a solution ' + 'less than or equal to amax: %s' % amax
            warn(msg, LineSearchWarning, stacklevel=2)
            break
        not_first_iteration = i > 0
        if phi_a1 > phi0 + c1 * alpha1 * derphi0 or (phi_a1 >= phi_a0 and not_first_iteration):
            alpha_star, phi_star, derphi_star = _zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi, derphi, phi0, derphi0, c1, c2, extra_condition)
            break
        derphi_a1 = derphi(alpha1)
        if abs(derphi_a1) <= -c2 * derphi0:
            if extra_condition(alpha1, phi_a1):
                alpha_star = alpha1
                phi_star = phi_a1
                derphi_star = derphi_a1
                break
        if derphi_a1 >= 0:
            alpha_star, phi_star, derphi_star = _zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi, derphi, phi0, derphi0, c1, c2, extra_condition)
            break
        alpha2 = 2 * alpha1
        if amax is not None:
            alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1
    else:
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None
        warn('The line search algorithm did not converge', LineSearchWarning, stacklevel=2)
    return (alpha_star, phi_star, phi0, derphi_star)