import re
import warnings
from enum import Enum
from math import gcd
def as_complex(real, imag=0):
    """Return object as COMPLEX expression (complex literal constant).
    """
    return Expr(Op.COMPLEX, (as_expr(real), as_expr(imag)))