import math
from .errors import Error as Cu2QuError, ApproxNotFoundError
@cython.cfunc
@cython.inline
@cython.returns(cython.complex)
@cython.locals(a=cython.complex, b=cython.complex, c=cython.complex, d=cython.complex)
@cython.locals(ab=cython.complex, cd=cython.complex, p=cython.complex, h=cython.double)
def calc_intersect(a, b, c, d):
    """Calculate the intersection of two lines.

    Args:
        a (complex): Start point of first line.
        b (complex): End point of first line.
        c (complex): Start point of second line.
        d (complex): End point of second line.

    Returns:
        complex: Location of intersection if one present, ``complex(NaN,NaN)``
        if no intersection was found.
    """
    ab = b - a
    cd = d - c
    p = ab * 1j
    try:
        h = dot(p, a - c) / dot(p, cd)
    except ZeroDivisionError:
        return complex(NAN, NAN)
    return c + cd * h