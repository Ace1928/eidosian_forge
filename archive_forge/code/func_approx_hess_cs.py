import numpy as np
from statsmodels.compat.pandas import Appender, Substitution
def approx_hess_cs(x, f, epsilon=None, args=(), kwargs={}):
    """Calculate Hessian with complex-step derivative approximation

    Parameters
    ----------
    x : array_like
       value at which function derivative is evaluated
    f : function
       function of one array f(x)
    epsilon : float
       stepsize, if None, then stepsize is automatically chosen

    Returns
    -------
    hess : ndarray
       array of partial second derivatives, Hessian

    Notes
    -----
    based on equation 10 in
    M. S. RIDOUT: Statistical Applications of the Complex-step Method
    of Numerical Differentiation, University of Kent, Canterbury, Kent, U.K.

    The stepsize is the same for the complex and the finite difference part.
    """
    n = len(x)
    h = _get_epsilon(x, 3, epsilon, n)
    ee = np.diag(h)
    hess = np.outer(h, h)
    n = len(x)
    for i in range(n):
        for j in range(i, n):
            hess[i, j] = np.squeeze((f(*(x + 1j * ee[i, :] + ee[j, :],) + args, **kwargs) - f(*(x + 1j * ee[i, :] - ee[j, :],) + args, **kwargs)).imag / 2.0 / hess[i, j])
            hess[j, i] = hess[i, j]
    return hess