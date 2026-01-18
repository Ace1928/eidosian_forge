from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer, Tuple,
from sympy.integrals import Integral
from sympy.concrete import Sum
from sympy.functions import (exp, sin, cos, fresnelc, fresnels, conjugate, Max,
from sympy.printing.mathematica import mathematica_code as mcode
def test_Integral():
    assert mcode(Integral(sin(sin(x)), x)) == 'Hold[Integrate[Sin[Sin[x]], x]]'
    assert mcode(Integral(exp(-x ** 2 - y ** 2), (x, -oo, oo), (y, -oo, oo))) == 'Hold[Integrate[Exp[-x^2 - y^2], {x, -Infinity, Infinity}, {y, -Infinity, Infinity}]]'