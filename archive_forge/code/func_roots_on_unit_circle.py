import spherogram
import snappy
import numpy as np
import mpmath
from sage.all import PolynomialRing, LaurentPolynomialRing, RR, ZZ, RealField, ComplexField, matrix, arccos, exp
def roots_on_unit_circle(poly, prec=53):
    """
    For a palindromic polynomial p(x) of even degree, return all the
    roots on the unit circle in the form

        (argument/(2 pi), multiplicity)

    """
    assert is_palindromic(poly) and poly.degree() % 2 == 0
    assert poly.parent().is_exact()
    assert poly(1) != 0 and poly(-1) != 0
    ans = []
    RR = RealField(prec)
    pi = RR.pi()
    g = compact_form(poly)
    for f, e in g.factor():
        roots = [r for r in f.roots(RR, False) if -2 < r < 2]
        args = [arccos(r / 2) / (2 * pi) for r in roots]
        args += [1 - a for a in args]
        ans += [(arg, e) for arg in args]
    return sorted(ans)