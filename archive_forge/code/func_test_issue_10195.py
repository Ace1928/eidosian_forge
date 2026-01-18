from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (E, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import sin
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.simplify.powsimp import (powdenest, powsimp)
from sympy.simplify.simplify import (signsimp, simplify)
from sympy.core.symbol import Str
from sympy.abc import x, y, z, a, b
def test_issue_10195():
    a = Symbol('a', integer=True)
    l = Symbol('l', even=True, nonzero=True)
    n = Symbol('n', odd=True)
    e_x = (-1) ** (n / 2 - S.Half) - (-1) ** (n * Rational(3, 2) - S.Half)
    assert powsimp((-1) ** (l / 2)) == I ** l
    assert powsimp((-1) ** (n / 2)) == I ** n
    assert powsimp((-1) ** (n * Rational(3, 2))) == -I ** n
    assert powsimp(e_x) == (-1) ** (n / 2 - S.Half) + (-1) ** (n * Rational(3, 2) + S.Half)
    assert powsimp((-1) ** (a * Rational(3, 2))) == (-I) ** a