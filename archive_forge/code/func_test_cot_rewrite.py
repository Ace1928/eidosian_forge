from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.function import (Lambda, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, atan2,
from sympy.functions.special.bessel import (besselj, jn)
from sympy.functions.special.delta_functions import Heaviside
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (cancel, gcd)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.series.series import series
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.core.relational import Ne, Eq
from sympy.functions.elementary.piecewise import Piecewise
from sympy.sets.setexpr import SetExpr
from sympy.testing.pytest import XFAIL, slow, raises
def test_cot_rewrite():
    neg_exp, pos_exp = (exp(-x * I), exp(x * I))
    assert cot(x).rewrite(exp) == I * (pos_exp + neg_exp) / (pos_exp - neg_exp)
    assert cot(x).rewrite(sin) == sin(2 * x) / (2 * sin(x) ** 2)
    assert cot(x).rewrite(cos) == cos(x) / cos(x - pi / 2, evaluate=False)
    assert cot(x).rewrite(tan) == 1 / tan(x)

    def check(func):
        z = cot(func(x)).rewrite(exp) - cot(x).rewrite(exp).subs(x, func(x))
        assert z.rewrite(exp).expand() == 0
    check(sinh)
    check(cosh)
    check(tanh)
    check(coth)
    check(sin)
    check(cos)
    check(tan)
    assert cot(log(x)).rewrite(Pow) == -I * (x ** (-I) + x ** I) / (x ** (-I) - x ** I)
    assert cot(x).rewrite(sec) == sec(x - pi / 2, evaluate=False) / sec(x)
    assert cot(x).rewrite(csc) == csc(x) / csc(-x + pi / 2, evaluate=False)
    assert cot(sin(x)).rewrite(Pow) == cot(sin(x))