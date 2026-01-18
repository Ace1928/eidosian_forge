from warnings import warn
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult
def create_spline(y, yp, x, h):
    """Create a cubic spline given values and derivatives.

    Formulas for the coefficients are taken from interpolate.CubicSpline.

    Returns
    -------
    sol : PPoly
        Constructed spline as a PPoly instance.
    """
    from scipy.interpolate import PPoly
    n, m = y.shape
    c = np.empty((4, n, m - 1), dtype=y.dtype)
    slope = (y[:, 1:] - y[:, :-1]) / h
    t = (yp[:, :-1] + yp[:, 1:] - 2 * slope) / h
    c[0] = t / h
    c[1] = (slope - yp[:, :-1]) / h - t
    c[2] = yp[:, :-1]
    c[3] = y[:, :-1]
    c = np.moveaxis(c, 1, 0)
    return PPoly(c, x, extrapolate=True, axis=1)