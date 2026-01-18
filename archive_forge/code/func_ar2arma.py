from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
def ar2arma(ar_des, p, q, n=20, mse='ar', start=None):
    """
    Find arma approximation to ar process.

    This finds the ARMA(p,q) coefficients that minimize the integrated
    squared difference between the impulse_response functions (MA
    representation) of the AR and the ARMA process. This does not  check
    whether the MA lag polynomial of the ARMA process is invertible, neither
    does it check the roots of the AR lag polynomial.

    Parameters
    ----------
    ar_des : array_like
        The coefficients of original AR lag polynomial, including lag zero.
    p : int
        The length of desired AR lag polynomials.
    q : int
        The length of desired MA lag polynomials.
    n : int
        The number of terms of the impulse_response function to include in the
        objective function for the approximation.
    mse : str, 'ar'
        Not used.
    start : ndarray
        Initial values to use when finding the approximation.

    Returns
    -------
    ar_app : ndarray
        The coefficients of the AR lag polynomials of the approximation.
    ma_app : ndarray
        The coefficients of the MA lag polynomials of the approximation.
    res : tuple
        The result of optimize.leastsq.

    Notes
    -----
    Extension is possible if we want to match autocovariance instead
    of impulse response function.
    """

    def msear_err(arma, ar_des):
        ar, ma = (np.r_[1, arma[:p - 1]], np.r_[1, arma[p - 1:]])
        ar_approx = arma_impulse_response(ma, ar, n)
        return ar_des - ar_approx
    if start is None:
        arma0 = np.r_[-0.9 * np.ones(p - 1), np.zeros(q - 1)]
    else:
        arma0 = start
    res = optimize.leastsq(msear_err, arma0, ar_des, maxfev=5000)
    arma_app = np.atleast_1d(res[0])
    ar_app = (np.r_[1, arma_app[:p - 1]],)
    ma_app = np.r_[1, arma_app[p - 1:]]
    return (ar_app, ma_app, res)