from sympy.core.function import diff
from sympy.core.numbers import (I, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, cot, sin)
from sympy.functions.special.spherical_harmonics import Ynm, Znm, Ynm_c
def test_Ynm_c():
    th, ph = (Symbol('theta', real=True), Symbol('phi', real=True))
    from sympy.abc import n, m
    assert Ynm_c(n, m, th, ph) == (-1) ** (2 * m) * exp(-2 * I * m * ph) * Ynm(n, m, th, ph)