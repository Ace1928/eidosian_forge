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
def gaunt_ref(l1, l2, l3, m1, m2, m3):
    return sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4 * pi)) * wigner_3j(l1, l2, l3, 0, 0, 0) * wigner_3j(l1, l2, l3, m1, m2, m3)