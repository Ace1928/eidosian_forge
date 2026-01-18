from sympy.core.numbers import (AlgebraicNumber, I, pi, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.external.gmpy import MPQ
from sympy.polys.numberfields.subfield import (
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.testing.pytest import raises
from sympy.abc import x
def test_issue_22736():
    a = CRootOf(x ** 4 + x ** 3 + x ** 2 + x + 1, -1)
    a._reset()
    b = exp(2 * I * pi / 5)
    assert field_isomorphism(a, b) == [1, 0]