from sympy.core.numbers import (I, pi, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import Matrix
from sympy.physics.wigner import (clebsch_gordan, wigner_9j, wigner_6j, gaunt,
from sympy.testing.pytest import raises
def test_clebsch_gordan_docs():
    assert clebsch_gordan(Rational(3, 2), S.Half, 2, Rational(3, 2), S.Half, 2) == 1
    assert clebsch_gordan(Rational(3, 2), S.Half, 1, Rational(3, 2), Rational(-1, 2), 1) == sqrt(3) / 2
    assert clebsch_gordan(Rational(3, 2), S.Half, 1, Rational(-1, 2), S.Half, 0) == -sqrt(2) / 2