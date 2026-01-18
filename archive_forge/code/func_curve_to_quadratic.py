import math
from .errors import Error as Cu2QuError, ApproxNotFoundError
@cython.locals(max_err=cython.double)
@cython.locals(n=cython.int)
@cython.locals(all_quadratic=cython.int)
def curve_to_quadratic(curve, max_err, all_quadratic=True):
    """Approximate a cubic Bezier curve with a spline of n quadratics.

    Args:
        cubic (sequence): Four 2D tuples representing control points of
            the cubic Bezier curve.
        max_err (double): Permitted deviation from the original curve.
        all_quadratic (bool): If True (default) returned value is a
            quadratic spline. If False, it's either a single quadratic
            curve or a single cubic curve.

    Returns:
        If all_quadratic is True: A list of 2D tuples, representing
        control points of the quadratic spline if it fits within the
        given tolerance, or ``None`` if no suitable spline could be
        calculated.

        If all_quadratic is False: Either a quadratic curve (if length
        of output is 3), or a cubic curve (if length of output is 4).
    """
    curve = [complex(*p) for p in curve]
    for n in range(1, MAX_N + 1):
        spline = cubic_approx_spline(curve, n, max_err, all_quadratic)
        if spline is not None:
            return [(s.real, s.imag) for s in spline]
    raise ApproxNotFoundError(curve)