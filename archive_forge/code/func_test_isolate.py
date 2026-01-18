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
def test_isolate():
    assert isolate(1) == (1, 1)
    assert isolate(S.Half) == (S.Half, S.Half)
    assert isolate(sqrt(2)) == (1, 2)
    assert isolate(-sqrt(2)) == (-2, -1)
    assert isolate(sqrt(2), eps=Rational(1, 100)) == (Rational(24, 17), Rational(17, 12))
    assert isolate(-sqrt(2), eps=Rational(1, 100)) == (Rational(-17, 12), Rational(-24, 17))
    raises(NotImplementedError, lambda: isolate(I))