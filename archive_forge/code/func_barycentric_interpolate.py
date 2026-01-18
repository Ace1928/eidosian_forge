import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
def barycentric_interpolate(xi, yi, x, axis=0):
    """Convenience function for polynomial interpolation.

    Constructs a polynomial that passes through a given
    set of points, then evaluates the polynomial. For
    reasons of numerical stability, this function does
    not compute the coefficients of the polynomial.

    Parameters
    ----------
    xi : cupy.ndarray
        1-D array of coordinates of the points the polynomial
        should pass through
    yi : cupy.ndarray
        y-coordinates of the points the polynomial should pass
        through
    x : scalar or cupy.ndarray
        Points to evaluate the interpolator at
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate
        values

    Returns
    -------
    y : scalar or cupy.ndarray
        Interpolated values. Shape is determined by replacing
        the interpolation axis in the original array with the
        shape x

    See Also
    --------
    scipy.interpolate.barycentric_interpolate

    """
    return BarycentricInterpolator(xi, yi, axis=axis)(x)