from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import (Derivative, Function, Lambda, Subs)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.core.sympify import SympifyError
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan2, cos, cot, sin, tan)
from sympy.matrices.dense import (Matrix, zeros)
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import RootOf
from sympy.simplify.cse_main import cse
from sympy.simplify.simplify import nsimplify
from sympy.core.basic import _aresame
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import a, x, y, z, t
def test_issue_15234():
    x, y = symbols('x y', real=True)
    p = 6 * x ** 5 + x ** 4 - 4 * x ** 3 + 4 * x ** 2 - 2 * x + 3
    p_subbed = 6 * x ** 5 - 4 * x ** 3 - 2 * x + y ** 4 + 4 * y ** 2 + 3
    assert p.subs([(x ** i, y ** i) for i in [2, 4]]) == p_subbed
    x, y = symbols('x y', complex=True)
    p = 6 * x ** 5 + x ** 4 - 4 * x ** 3 + 4 * x ** 2 - 2 * x + 3
    p_subbed = 6 * x ** 5 - 4 * x ** 3 - 2 * x + y ** 4 + 4 * y ** 2 + 3
    assert p.subs([(x ** i, y ** i) for i in [2, 4]]) == p_subbed