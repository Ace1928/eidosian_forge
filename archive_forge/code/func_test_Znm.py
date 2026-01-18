from sympy.core.function import diff
from sympy.core.numbers import (I, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, cot, sin)
from sympy.functions.special.spherical_harmonics import Ynm, Znm, Ynm_c
def test_Znm():
    th, ph = (Symbol('theta', real=True), Symbol('phi', real=True))
    assert Znm(0, 0, th, ph) == Ynm(0, 0, th, ph)
    assert Znm(1, -1, th, ph) == -sqrt(2) * I * (Ynm(1, 1, th, ph) - exp(-2 * I * ph) * Ynm(1, 1, th, ph)) / 2
    assert Znm(1, 0, th, ph) == Ynm(1, 0, th, ph)
    assert Znm(1, 1, th, ph) == sqrt(2) * (Ynm(1, 1, th, ph) + exp(-2 * I * ph) * Ynm(1, 1, th, ph)) / 2
    assert Znm(0, 0, th, ph).expand(func=True) == 1 / (2 * sqrt(pi))
    assert Znm(1, -1, th, ph).expand(func=True) == sqrt(3) * I * sin(th) * exp(I * ph) / (4 * sqrt(pi)) - sqrt(3) * I * sin(th) * exp(-I * ph) / (4 * sqrt(pi))
    assert Znm(1, 0, th, ph).expand(func=True) == sqrt(3) * cos(th) / (2 * sqrt(pi))
    assert Znm(1, 1, th, ph).expand(func=True) == -sqrt(3) * sin(th) * exp(I * ph) / (4 * sqrt(pi)) - sqrt(3) * sin(th) * exp(-I * ph) / (4 * sqrt(pi))
    assert Znm(2, -1, th, ph).expand(func=True) == sqrt(15) * I * sin(th) * exp(I * ph) * cos(th) / (4 * sqrt(pi)) - sqrt(15) * I * sin(th) * exp(-I * ph) * cos(th) / (4 * sqrt(pi))
    assert Znm(2, 0, th, ph).expand(func=True) == 3 * sqrt(5) * cos(th) ** 2 / (4 * sqrt(pi)) - sqrt(5) / (4 * sqrt(pi))
    assert Znm(2, 1, th, ph).expand(func=True) == -sqrt(15) * sin(th) * exp(I * ph) * cos(th) / (4 * sqrt(pi)) - sqrt(15) * sin(th) * exp(-I * ph) * cos(th) / (4 * sqrt(pi))