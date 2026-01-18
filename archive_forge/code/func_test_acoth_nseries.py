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
def test_acoth_nseries():
    x = Symbol('x')
    assert acoth(x + 1)._eval_nseries(x, 4, None) == log(2) / 2 - log(x) / 2 + x / 4 - x ** 2 / 16 + x ** 3 / 48 + O(x ** 4)
    assert acoth(x - 1)._eval_nseries(x, 4, None, cdir=1) == I * pi / 2 - log(2) / 2 + log(x) / 2 + x / 4 + x ** 2 / 16 + x ** 3 / 48 + O(x ** 4)
    assert acoth(x - 1)._eval_nseries(x, 4, None, cdir=-1) == -I * pi / 2 - log(2) / 2 + log(x) / 2 + x / 4 + x ** 2 / 16 + x ** 3 / 48 + O(x ** 4)
    assert acoth(I * x + S(1) / 2)._eval_nseries(x, 4, None, cdir=1) == acoth(S(1) / 2) + 4 * I * x / 3 - 8 * x ** 2 / 9 - 112 * I * x ** 3 / 81 + O(x ** 4)
    assert acoth(I * x + S(1) / 2)._eval_nseries(x, 4, None, cdir=-1) == I * pi + acoth(S(1) / 2) + 4 * I * x / 3 - 8 * x ** 2 / 9 - 112 * I * x ** 3 / 81 + O(x ** 4)
    assert acoth(I * x - S(1) / 2)._eval_nseries(x, 4, None, cdir=1) == -acoth(S(1) / 2) - I * pi + 4 * I * x / 3 + 8 * x ** 2 / 9 - 112 * I * x ** 3 / 81 + O(x ** 4)
    assert acoth(I * x - S(1) / 2)._eval_nseries(x, 4, None, cdir=-1) == -acoth(S(1) / 2) + 4 * I * x / 3 + 8 * x ** 2 / 9 - 112 * I * x ** 3 / 81 + O(x ** 4)
    assert acoth(-I * x ** 2 - x - S(1) / 2)._eval_nseries(x, 4, None) == I * pi / 2 - log(3) / 2 - 4 * x / 3 + x ** 2 * (-S(8) / 9 + 2 * I / 3) - 2 * I * x ** 2 + x ** 3 * (S(104) / 81 - 16 * I / 9) - 8 * x ** 3 / 3 + O(x ** 4)