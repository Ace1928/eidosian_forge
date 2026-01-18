import functools
import warnings
import numpy
import cupy
import cupyx.scipy.fft
def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """Returns the least squares fit of polynomial of degree deg
    to the data y sampled at x.

    Args:
        x (cupy.ndarray): x-coordinates of the sample points of shape (M,).
        y (cupy.ndarray): y-coordinates of the sample points of shape
            (M,) or (M, K).
        deg (int): degree of the fitting polynomial.
        rcond (float, optional): relative condition number of the fit.
            The default value is ``len(x) * eps``.
        full (bool, optional): indicator of the return value nature.
            When False (default), only the coefficients are returned.
            When True, diagnostic information is also returned.
        w (cupy.ndarray, optional): weights applied to the y-coordinates
            of the sample points of shape (M,).
        cov (bool or str, optional): if given, returns the coefficients
            along with the covariance matrix.

    Returns:
        cupy.ndarray or tuple:
        p (cupy.ndarray of shape (deg + 1,) or (deg + 1, K)):
            Polynomial coefficients from highest to lowest degree
        residuals, rank, singular_values, rcond         (cupy.ndarray, int, cupy.ndarray, float):
            Present only if ``full=True``.
            Sum of squared residuals of the least-squares fit,
            rank of the scaled Vandermonde coefficient matrix,
            its singular values, and the specified value of ``rcond``.
        V (cupy.ndarray of shape (M, M) or (M, M, K)):
            Present only if ``full=False`` and ``cov=True``.
            The covariance matrix of the polynomial coefficient estimates.

    .. warning::

        numpy.RankWarning: The rank of the coefficient matrix in the
        least-squares fit is deficient. It is raised if ``full=False``.

    .. seealso:: :func:`numpy.polyfit`

    """
    if x.dtype.char == 'e' and y.dtype.kind == 'b':
        raise NotImplementedError('float16 x and bool y are not currently supported')
    if y.dtype == numpy.float16:
        raise TypeError('float16 y are not supported')
    x = _polyfit_typecast(x)
    y = _polyfit_typecast(y)
    deg = int(deg)
    if deg < 0:
        raise ValueError('expected deg >= 0')
    if x.ndim != 1:
        raise TypeError('expected 1D vector for x')
    if x.size == 0:
        raise TypeError('expected non-empty vector for x')
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError('expected 1D or 2D array for y')
    if x.size != y.shape[0]:
        raise TypeError('expected x and y to have same length')
    lhs = cupy.polynomial.polynomial.polyvander(x, deg)[:, ::-1]
    rhs = y
    if w is not None:
        w = _polyfit_typecast(w)
        if w.ndim != 1:
            raise TypeError('expected a 1-d array for weights')
        if w.size != x.size:
            raise TypeError('expected w and y to have the same length')
        lhs *= w[:, None]
        if rhs.ndim == 2:
            w = w[:, None]
        rhs *= w
    if rcond is None:
        rcond = x.size * cupy.finfo(x.dtype).eps
    scale = cupy.sqrt(cupy.square(lhs).sum(axis=0))
    lhs /= scale
    c, resids, rank, s = cupy.linalg.lstsq(lhs, rhs, rcond)
    if y.ndim > 1:
        scale = scale.reshape(-1, 1)
    c /= scale
    order = deg + 1
    if rank != order and (not full):
        msg = 'Polyfit may be poorly conditioned'
        warnings.warn(msg, numpy.RankWarning, stacklevel=4)
    if full:
        if resids.dtype.kind == 'c':
            resids = cupy.absolute(resids)
        return (c, resids, rank, s, rcond)
    if cov:
        base = cupy.linalg.inv(cupy.dot(lhs.T, lhs))
        base /= cupy.outer(scale, scale)
        if cov == 'unscaled':
            factor = 1
        elif x.size > order:
            factor = resids / (x.size - order)
        else:
            raise ValueError('the number of data points must exceed order to scale the covariance matrix')
        if y.ndim != 1:
            base = base[..., None]
        return (c, base * factor)
    return c