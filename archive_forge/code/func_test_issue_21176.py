from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cot, sin, tan)
from sympy.series.residues import residue
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, z, a, s, k
def test_issue_21176():
    f = x ** 2 * cot(pi * x) / (x ** 4 + 1)
    assert residue(f, x, -sqrt(2) / 2 - sqrt(2) * I / 2).cancel().together(deep=True) == sqrt(2) * (1 - I) / (8 * tan(sqrt(2) * pi * (1 + I) / 2))