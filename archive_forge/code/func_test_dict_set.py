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
def test_dict_set():
    a, b, c = map(Wild, 'abc')
    f = 3 * cos(4 * x)
    r = f.match(a * cos(b * x))
    assert r == {a: 3, b: 4}
    e = a / b * sin(b * x)
    assert e.subs(r) == r[a] / r[b] * sin(r[b] * x)
    assert e.subs(r) == 3 * sin(4 * x) / 4
    s = set(r.items())
    assert e.subs(s) == r[a] / r[b] * sin(r[b] * x)
    assert e.subs(s) == 3 * sin(4 * x) / 4
    assert e.subs(r) == r[a] / r[b] * sin(r[b] * x)
    assert e.subs(r) == 3 * sin(4 * x) / 4
    assert x.subs(Dict((x, 1))) == 1