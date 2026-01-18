import math
from .errors import Error as Cu2QuError, ApproxNotFoundError
@cython.cfunc
@cython.locals(n=cython.int, tolerance=cython.double)
@cython.locals(i=cython.int)
@cython.locals(all_quadratic=cython.int)
@cython.locals(c0=cython.complex, c1=cython.complex, c2=cython.complex, c3=cython.complex)
@cython.locals(q0=cython.complex, q1=cython.complex, next_q1=cython.complex, q2=cython.complex, d1=cython.complex)
def cubic_approx_spline(cubic, n, tolerance, all_quadratic):
    """Approximate a cubic Bezier curve with a spline of n quadratics.

    Args:
        cubic (sequence): Four complex numbers representing control points of
            the cubic Bezier curve.
        n (int): Number of quadratic Bezier curves in the spline.
        tolerance (double): Permitted deviation from the original curve.

    Returns:
        A list of ``n+2`` complex numbers, representing control points of the
        quadratic spline if it fits within the given tolerance, or ``None`` if
        no suitable spline could be calculated.
    """
    if n == 1:
        return cubic_approx_quadratic(cubic, tolerance)
    if n == 2 and all_quadratic == False:
        return cubic
    cubics = split_cubic_into_n_iter(cubic[0], cubic[1], cubic[2], cubic[3], n)
    next_cubic = next(cubics)
    next_q1 = cubic_approx_control(0, next_cubic[0], next_cubic[1], next_cubic[2], next_cubic[3])
    q2 = cubic[0]
    d1 = 0j
    spline = [cubic[0], next_q1]
    for i in range(1, n + 1):
        c0, c1, c2, c3 = next_cubic
        q0 = q2
        q1 = next_q1
        if i < n:
            next_cubic = next(cubics)
            next_q1 = cubic_approx_control(i / (n - 1), next_cubic[0], next_cubic[1], next_cubic[2], next_cubic[3])
            spline.append(next_q1)
            q2 = (q1 + next_q1) * 0.5
        else:
            q2 = c3
        d0 = d1
        d1 = q2 - c3
        if abs(d1) > tolerance or not cubic_farthest_fit_inside(d0, q0 + (q1 - q0) * (2 / 3) - c1, q2 + (q1 - q2) * (2 / 3) - c2, d1, tolerance):
            return None
    spline.append(cubic[3])
    return spline