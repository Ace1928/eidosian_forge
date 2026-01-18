import math
from .errors import Error as Cu2QuError, ApproxNotFoundError
@cython.cfunc
@cython.inline
@cython.locals(p0=cython.complex, p1=cython.complex, p2=cython.complex, p3=cython.complex)
@cython.locals(mid=cython.complex, deriv3=cython.complex)
def split_cubic_into_two(p0, p1, p2, p3):
    """Split a cubic Bezier into two equal parts.

    Splits the curve into two equal parts at t = 0.5

    Args:
        p0 (complex): Start point of curve.
        p1 (complex): First handle of curve.
        p2 (complex): Second handle of curve.
        p3 (complex): End point of curve.

    Returns:
        tuple: Two cubic Beziers (each expressed as a tuple of four complex
        values).
    """
    mid = (p0 + 3 * (p1 + p2) + p3) * 0.125
    deriv3 = (p3 + p2 - p1 - p0) * 0.125
    return ((p0, (p0 + p1) * 0.5, mid - deriv3, mid), (mid, mid + deriv3, (p2 + p3) * 0.5, p3))