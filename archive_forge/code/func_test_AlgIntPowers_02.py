from sympy.abc import x
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.matrices.exceptions import DMRankError
from sympy.polys.numberfields.utilities import (
from sympy.printing.lambdarepr import IntervalPrinter
from sympy.testing.pytest import raises
def test_AlgIntPowers_02():
    T = Poly(x ** 3 + 2 * x ** 2 + 3 * x + 4)
    m = 7
    theta_pow = AlgIntPowers(T, m)
    for e in range(10):
        computed = theta_pow[e]
        coeffs = (Poly(x) ** e % T + Poly(x ** 3)).rep.rep[1:]
        expected = [c % m for c in reversed(coeffs)]
        assert computed == expected