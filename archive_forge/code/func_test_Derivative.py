from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer, Tuple,
from sympy.integrals import Integral
from sympy.concrete import Sum
from sympy.functions import (exp, sin, cos, fresnelc, fresnels, conjugate, Max,
from sympy.printing.mathematica import mathematica_code as mcode
def test_Derivative():
    assert mcode(Derivative(sin(x), x)) == 'Hold[D[Sin[x], x]]'
    assert mcode(Derivative(x, x)) == 'Hold[D[x, x]]'
    assert mcode(Derivative(sin(x) * y ** 4, x, 2)) == 'Hold[D[y^4*Sin[x], {x, 2}]]'
    assert mcode(Derivative(sin(x) * y ** 4, x, y, x)) == 'Hold[D[y^4*Sin[x], x, y, x]]'
    assert mcode(Derivative(sin(x) * y ** 4, x, y, 3, x)) == 'Hold[D[y^4*Sin[x], x, {y, 3}, x]]'