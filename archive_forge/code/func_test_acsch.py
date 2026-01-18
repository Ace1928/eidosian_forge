from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.function import (expand_mul, expand_trig)
from sympy.core.numbers import (E, I, Integer, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, acsch, asech, asinh, atanh, cosh, coth, csch, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, cos, cot, sec, sin, tan)
from sympy.series.order import O
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises
def test_acsch():
    x = Symbol('x')
    assert unchanged(acsch, x)
    assert acsch(-x) == -acsch(x)
    assert acsch(1) == log(1 + sqrt(2))
    assert acsch(-1) == -log(1 + sqrt(2))
    assert acsch(0) is zoo
    assert acsch(2) == log((1 + sqrt(5)) / 2)
    assert acsch(-2) == -log((1 + sqrt(5)) / 2)
    assert acsch(I) == -I * pi / 2
    assert acsch(-I) == I * pi / 2
    assert acsch(-I * (sqrt(6) + sqrt(2))) == I * pi / 12
    assert acsch(I * (sqrt(2) + sqrt(6))) == -I * pi / 12
    assert acsch(-I * (1 + sqrt(5))) == I * pi / 10
    assert acsch(I * (1 + sqrt(5))) == -I * pi / 10
    assert acsch(-I * 2 / sqrt(2 - sqrt(2))) == I * pi / 8
    assert acsch(I * 2 / sqrt(2 - sqrt(2))) == -I * pi / 8
    assert acsch(-I * 2) == I * pi / 6
    assert acsch(I * 2) == -I * pi / 6
    assert acsch(-I * sqrt(2 + 2 / sqrt(5))) == I * pi / 5
    assert acsch(I * sqrt(2 + 2 / sqrt(5))) == -I * pi / 5
    assert acsch(-I * sqrt(2)) == I * pi / 4
    assert acsch(I * sqrt(2)) == -I * pi / 4
    assert acsch(-I * (sqrt(5) - 1)) == 3 * I * pi / 10
    assert acsch(I * (sqrt(5) - 1)) == -3 * I * pi / 10
    assert acsch(-I * 2 / sqrt(3)) == I * pi / 3
    assert acsch(I * 2 / sqrt(3)) == -I * pi / 3
    assert acsch(-I * 2 / sqrt(2 + sqrt(2))) == 3 * I * pi / 8
    assert acsch(I * 2 / sqrt(2 + sqrt(2))) == -3 * I * pi / 8
    assert acsch(-I * sqrt(2 - 2 / sqrt(5))) == 2 * I * pi / 5
    assert acsch(I * sqrt(2 - 2 / sqrt(5))) == -2 * I * pi / 5
    assert acsch(-I * (sqrt(6) - sqrt(2))) == 5 * I * pi / 12
    assert acsch(I * (sqrt(6) - sqrt(2))) == -5 * I * pi / 12
    assert acsch(nan) is nan
    assert acsch(-I * sqrt(2)) == asinh(I / sqrt(2))
    assert acsch(-I * 2 / sqrt(3)) == asinh(I * sqrt(3) / 2)
    assert acsch(-I * sqrt(2)) == -I * asin(-1 / sqrt(2))
    assert acsch(-I * 2 / sqrt(3)) == -I * asin(-sqrt(3) / 2)
    assert expand_mul(csch(acsch(-I * (sqrt(6) + sqrt(2)))) / (-I * (sqrt(6) + sqrt(2)))) == 1
    assert expand_mul(csch(acsch(I * (1 + sqrt(5)))) / (I * (1 + sqrt(5)))) == 1
    assert (csch(acsch(I * sqrt(2 - 2 / sqrt(5)))) / (I * sqrt(2 - 2 / sqrt(5)))).simplify() == 1
    assert (csch(acsch(-I * sqrt(2 - 2 / sqrt(5)))) / (-I * sqrt(2 - 2 / sqrt(5)))).simplify() == 1
    assert str(acsch(5 * I + 1).n(6)) == '0.0391819 - 0.193363*I'
    assert str(acsch(-5 * I + 1).n(6)) == '0.0391819 + 0.193363*I'