import math
from .errors import Error as Cu2QuError, ApproxNotFoundError
@cython.cfunc
@cython.inline
@cython.locals(tolerance=cython.double)
@cython.locals(q1=cython.complex, c0=cython.complex, c1=cython.complex, c2=cython.complex, c3=cython.complex)
def cubic_approx_quadratic(cubic, tolerance):
    """Approximate a cubic Bezier with a single quadratic within a given tolerance.

    Args:
        cubic (sequence): Four complex numbers representing control points of
            the cubic Bezier curve.
        tolerance (double): Permitted deviation from the original curve.

    Returns:
        Three complex numbers representing control points of the quadratic
        curve if it fits within the given tolerance, or ``None`` if no suitable
        curve could be calculated.
    """
    q1 = calc_intersect(cubic[0], cubic[1], cubic[2], cubic[3])
    if math.isnan(q1.imag):
        return None
    c0 = cubic[0]
    c3 = cubic[3]
    c1 = c0 + (q1 - c0) * (2 / 3)
    c2 = c3 + (q1 - c3) * (2 / 3)
    if not cubic_farthest_fit_inside(0, c1 - cubic[1], c2 - cubic[2], 0, tolerance):
        return None
    return (c0, q1, c3)