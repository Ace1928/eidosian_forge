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
def test_acsch_nseries():
    x = Symbol('x')
    assert acsch(x + I)._eval_nseries(x, 4, None) == -I * pi / 2 + I * sqrt(x) + sqrt(x) + 5 * I * x ** (S(3) / 2) / 12 - 5 * x ** (S(3) / 2) / 12 - 43 * I * x ** (S(5) / 2) / 160 - 43 * x ** (S(5) / 2) / 160 - 177 * I * x ** (S(7) / 2) / 896 + 177 * x ** (S(7) / 2) / 896 + O(x ** 4)
    assert acsch(x - I)._eval_nseries(x, 4, None) == I * pi / 2 - I * sqrt(x) + sqrt(x) - 5 * I * x ** (S(3) / 2) / 12 - 5 * x ** (S(3) / 2) / 12 + 43 * I * x ** (S(5) / 2) / 160 - 43 * x ** (S(5) / 2) / 160 + 177 * I * x ** (S(7) / 2) / 896 + 177 * x ** (S(7) / 2) / 896 + O(x ** 4)
    assert acsch(x + I / 2)._eval_nseries(x, 4, None, cdir=1) == -acsch(I / 2) - I * pi + 4 * sqrt(3) * I * x / 3 - 8 * sqrt(3) * x ** 2 / 9 - 16 * sqrt(3) * I * x ** 3 / 9 + O(x ** 4)
    assert acsch(x + I / 2)._eval_nseries(x, 4, None, cdir=-1) == acsch(I / 2) - 4 * sqrt(3) * I * x / 3 + 8 * sqrt(3) * x ** 2 / 9 + 16 * sqrt(3) * I * x ** 3 / 9 + O(x ** 4)
    assert acsch(x - I / 2)._eval_nseries(x, 4, None, cdir=1) == -acsch(I / 2) - 4 * sqrt(3) * I * x / 3 - 8 * sqrt(3) * x ** 2 / 9 + 16 * sqrt(3) * I * x ** 3 / 9 + O(x ** 4)
    assert acsch(x - I / 2)._eval_nseries(x, 4, None, cdir=-1) == I * pi + acsch(I / 2) + 4 * sqrt(3) * I * x / 3 + 8 * sqrt(3) * x ** 2 / 9 - 16 * sqrt(3) * I * x ** 3 / 9 + O(x ** 4)
    assert acsch(I / 2 + I * x - x ** 2)._eval_nseries(x, 4, None) == -I * pi / 2 + log(2 - sqrt(3)) + 4 * sqrt(3) * x / 3 + x ** 2 * (-8 * sqrt(3) / 9 + 4 * sqrt(3) * I / 3) + x ** 3 * (16 * sqrt(3) / 9 - 16 * sqrt(3) * I / 9) + O(x ** 4)