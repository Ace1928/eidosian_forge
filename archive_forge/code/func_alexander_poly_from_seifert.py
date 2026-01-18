import spherogram
import snappy
import numpy as np
import mpmath
from sage.all import PolynomialRing, LaurentPolynomialRing, RR, ZZ, RealField, ComplexField, matrix, arccos, exp
def alexander_poly_from_seifert(V):
    R = PolynomialRing(ZZ, 't')
    t = R.gen()
    poly = (t * V - V.transpose()).determinant()
    if poly.leading_coefficient() < 0:
        poly = -poly
    e = min(poly.exponents())
    if e > 0:
        poly = poly // t ** e
    return poly