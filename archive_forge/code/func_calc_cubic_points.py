import math
from .errors import Error as Cu2QuError, ApproxNotFoundError
@cython.cfunc
@cython.inline
@cython.locals(a=cython.complex, b=cython.complex, c=cython.complex, d=cython.complex)
@cython.locals(_1=cython.complex, _2=cython.complex, _3=cython.complex, _4=cython.complex)
def calc_cubic_points(a, b, c, d):
    _1 = d
    _2 = c / 3.0 + d
    _3 = (b + c) / 3.0 + _2
    _4 = a + d + c + b
    return (_1, _2, _3, _4)