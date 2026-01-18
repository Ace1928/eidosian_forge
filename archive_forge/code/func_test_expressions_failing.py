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
@XFAIL
def test_expressions_failing():
    n = Symbol('n', integer=True, positive=True)
    assert residue(exp(z) / (z - pi * I / 4 * a) ** n, z, I * pi * a) == exp(I * pi * a / 4) / factorial(n - 1)