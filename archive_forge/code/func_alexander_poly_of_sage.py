import spherogram
import snappy
from sage.all import Link as SageLink
from sage.all import LaurentPolynomialRing, PolynomialRing, ZZ, var
from sage.symbolic.ring import SymbolicRing
def alexander_poly_of_sage(knot):
    p = knot.alexander_polynomial()
    ans = p.polynomial_construction()[0]
    if ans.leading_coefficient() < 0:
        ans = -ans
    return ans