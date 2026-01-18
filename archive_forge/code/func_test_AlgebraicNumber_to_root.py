from sympy.core.containers import Tuple
from sympy.core.numbers import (AlgebraicNumber, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import Poly
from sympy.polys.numberfields.subfield import to_number_field
from sympy.polys.polyclasses import DMP
from sympy.polys.domains import QQ
from sympy.polys.rootoftools import CRootOf
from sympy.abc import x, y
def test_AlgebraicNumber_to_root():
    assert AlgebraicNumber(sqrt(2)).to_root() == sqrt(2)
    zeta5_squared = AlgebraicNumber(CRootOf(x ** 5 - 1, 4), coeffs=[1, 0, 0])
    assert zeta5_squared.to_root() == CRootOf(x ** 4 + x ** 3 + x ** 2 + x + 1, 1)
    zeta3_squared = AlgebraicNumber(CRootOf(x ** 3 - 1, 2), coeffs=[1, 0, 0])
    assert zeta3_squared.to_root() == -S(1) / 2 - sqrt(3) * I / 2
    assert zeta3_squared.to_root(radicals=False) == CRootOf(x ** 2 + x + 1, 0)