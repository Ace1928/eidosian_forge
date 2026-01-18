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
def test_issue_22546():
    p1, p2 = symbols('p1, p2', positive=True)
    ref = powsimp(p1 ** z / p2 ** z)
    e = z + 1
    ans = ref.subs(z, e)
    assert ans.is_Pow
    assert powsimp(p1 ** e / p2 ** e) == ans
    i = symbols('i', integer=True)
    ref = powsimp(x ** i / y ** i)
    e = i + 1
    ans = ref.subs(i, e)
    assert ans.is_Pow
    assert powsimp(x ** e / y ** e) == ans