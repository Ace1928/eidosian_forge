import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
def krogh_interpolate(xi, yi, x, der=0, axis=0):
    """Convenience function for polynomial interpolation

    Parameters
    ----------
    xi : cupy.ndarray
        x-coordinate
    yi : cupy.ndarray
        y-coordinates, of shape ``(xi.size, R)``. Interpreted as
        vectors of length R, or scalars if R=1
    x : cupy.ndarray
        Point or points at which to evaluate the derivatives
    der : int or list, optional
        How many derivatives to extract; None for all potentially
        nonzero derivatives (that is a number equal to the number
        of points), or a list of derivatives to extract. This number
        includes the function value as 0th derivative
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values

    Returns
    -------
    d : cupy.ndarray
        If the interpolator's values are R-D then the
        returned array will be the number of derivatives by N by R.
        If `x` is a scalar, the middle dimension will be dropped; if
        the `yi` are scalars then the last dimension will be dropped

    See Also
    --------
    scipy.interpolate.krogh_interpolate

    """
    P = KroghInterpolator(xi, yi, axis=axis)
    if der == 0:
        return P(x)
    elif _isscalar(der):
        return P.derivative(x, der=der)
    else:
        return P.derivatives(x, der=cupy.amax(der) + 1)[der]