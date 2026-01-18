import math
from .errors import Error as Cu2QuError, ApproxNotFoundError
@cython.cfunc
@cython.inline
@cython.locals(p0=cython.complex, p1=cython.complex, p2=cython.complex, p3=cython.complex)
@cython.locals(a=cython.complex, b=cython.complex, c=cython.complex, d=cython.complex)
def calc_cubic_parameters(p0, p1, p2, p3):
    c = (p1 - p0) * 3.0
    b = (p2 - p1) * 3.0 - c
    d = p0
    a = p3 - d - c - b
    return (a, b, c, d)