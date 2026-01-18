from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def normalize_polynomial(f):
    """
    Multiply by t^-n so that the constant term is nonzero.

       sage: t = PolynomialRing(ZZ, 't').gen()
       sage: normalize_polynomial(t**3 + 2*t**2)
       t + 2
    """
    e = min(f.exponents())
    t = f.parent().gen()
    return f // t ** e