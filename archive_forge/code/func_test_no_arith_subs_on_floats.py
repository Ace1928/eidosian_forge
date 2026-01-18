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
def test_no_arith_subs_on_floats():
    assert (x + 3).subs(x + 3, a) == a
    assert (x + 3).subs(x + 2, a) == a + 1
    assert (x + y + 3).subs(x + 3, a) == a + y
    assert (x + y + 3).subs(x + 2, a) == a + y + 1
    assert (x + 3.0).subs(x + 3.0, a) == a
    assert (x + 3.0).subs(x + 2.0, a) == x + 3.0
    assert (x + y + 3.0).subs(x + 3.0, a) == a + y
    assert (x + y + 3.0).subs(x + 2.0, a) == x + y + 3.0