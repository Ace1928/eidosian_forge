import math
from .errors import Error as Cu2QuError, ApproxNotFoundError
@cython.cfunc
@cython.inline
@cython.locals(p0=cython.complex, p1=cython.complex, p2=cython.complex, p3=cython.complex)
@cython.locals(mid1=cython.complex, deriv1=cython.complex, mid2=cython.complex, deriv2=cython.complex)
def split_cubic_into_three(p0, p1, p2, p3):
    """Split a cubic Bezier into three equal parts.

    Splits the curve into three equal parts at t = 1/3 and t = 2/3

    Args:
        p0 (complex): Start point of curve.
        p1 (complex): First handle of curve.
        p2 (complex): Second handle of curve.
        p3 (complex): End point of curve.

    Returns:
        tuple: Three cubic Beziers (each expressed as a tuple of four complex
        values).
    """
    mid1 = (8 * p0 + 12 * p1 + 6 * p2 + p3) * (1 / 27)
    deriv1 = (p3 + 3 * p2 - 4 * p0) * (1 / 27)
    mid2 = (p0 + 6 * p1 + 12 * p2 + 8 * p3) * (1 / 27)
    deriv2 = (4 * p3 - 3 * p1 - p0) * (1 / 27)
    return ((p0, (2 * p0 + p1) / 3.0, mid1 - deriv1, mid1), (mid1, mid1 + deriv1, mid2 - deriv2, mid2), (mid2, mid2 + deriv2, (p2 + 2 * p3) / 3.0, p3))