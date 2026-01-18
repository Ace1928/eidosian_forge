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
def test_AlgIntPowers_01():
    T = Poly(cyclotomic_poly(5))
    zeta_pow = AlgIntPowers(T)
    raises(ValueError, lambda: zeta_pow[-1])
    for e in range(10):
        a = e % 5
        if a < 4:
            c = zeta_pow[e]
            assert c[a] == 1 and all((c[i] == 0 for i in range(4) if i != a))
        else:
            assert zeta_pow[e] == [-1] * 4