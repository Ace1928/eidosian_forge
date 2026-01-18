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
def test_extract_fundamental_discriminant():
    raises(ValueError, lambda: extract_fundamental_discriminant(2))
    raises(ValueError, lambda: extract_fundamental_discriminant(3))
    cases = ((0, {}, {0: 1}), (1, {}, {}), (8, {2: 3}, {}), (-8, {2: 3, -1: 1}, {}), (12, {2: 2, 3: 1}, {}), (36, {}, {2: 1, 3: 1}), (45, {5: 1}, {3: 1}), (48, {2: 2, 3: 1}, {2: 1}), (1125, {5: 1}, {3: 1, 5: 1}))
    for a, D_expected, F_expected in cases:
        D, F = extract_fundamental_discriminant(a)
        assert D == D_expected
        assert F == F_expected