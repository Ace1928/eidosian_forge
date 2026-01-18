from sympy import abc
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.hyper import meijerg
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import signsimp
from sympy.testing.pytest import XFAIL
def test_mul_noncommutative():
    x, y = symbols('x y')
    A, B, C = symbols('A B C', commutative=False)
    u, v = symbols('u v', cls=Wild)
    w, z = symbols('w z', cls=Wild, commutative=False)
    assert (u * v).matches(x) in ({v: x, u: 1}, {u: x, v: 1})
    assert (u * v).matches(x * y) in ({v: y, u: x}, {u: y, v: x})
    assert (u * v).matches(A) is None
    assert (u * v).matches(A * B) is None
    assert (u * v).matches(x * A) is None
    assert (u * v).matches(x * y * A) is None
    assert (u * v).matches(x * A * B) is None
    assert (u * v).matches(x * y * A * B) is None
    assert (v * w).matches(x) is None
    assert (v * w).matches(x * y) is None
    assert (v * w).matches(A) == {w: A, v: 1}
    assert (v * w).matches(A * B) == {w: A * B, v: 1}
    assert (v * w).matches(x * A) == {w: A, v: x}
    assert (v * w).matches(x * y * A) == {w: A, v: x * y}
    assert (v * w).matches(x * A * B) == {w: A * B, v: x}
    assert (v * w).matches(x * y * A * B) == {w: A * B, v: x * y}
    assert (v * w).matches(-x) is None
    assert (v * w).matches(-x * y) is None
    assert (v * w).matches(-A) == {w: A, v: -1}
    assert (v * w).matches(-A * B) == {w: A * B, v: -1}
    assert (v * w).matches(-x * A) == {w: A, v: -x}
    assert (v * w).matches(-x * y * A) == {w: A, v: -x * y}
    assert (v * w).matches(-x * A * B) == {w: A * B, v: -x}
    assert (v * w).matches(-x * y * A * B) == {w: A * B, v: -x * y}
    assert (w * z).matches(x) is None
    assert (w * z).matches(x * y) is None
    assert (w * z).matches(A) is None
    assert (w * z).matches(A * B) == {w: A, z: B}
    assert (w * z).matches(B * A) == {w: B, z: A}
    assert (w * z).matches(A * B * C) in [{w: A, z: B * C}, {w: A * B, z: C}]
    assert (w * z).matches(x * A) is None
    assert (w * z).matches(x * y * A) is None
    assert (w * z).matches(x * A * B) is None
    assert (w * z).matches(x * y * A * B) is None
    assert (w * A).matches(A) is None
    assert (A * w * B).matches(A * B) is None
    assert (u * w * z).matches(x) is None
    assert (u * w * z).matches(x * y) is None
    assert (u * w * z).matches(A) is None
    assert (u * w * z).matches(A * B) == {u: 1, w: A, z: B}
    assert (u * w * z).matches(B * A) == {u: 1, w: B, z: A}
    assert (u * w * z).matches(x * A) is None
    assert (u * w * z).matches(x * y * A) is None
    assert (u * w * z).matches(x * A * B) == {u: x, w: A, z: B}
    assert (u * w * z).matches(x * B * A) == {u: x, w: B, z: A}
    assert (u * w * z).matches(x * y * A * B) == {u: x * y, w: A, z: B}
    assert (u * w * z).matches(x * y * B * A) == {u: x * y, w: B, z: A}
    assert (u * A).matches(x * A) == {u: x}
    assert (u * A).matches(x * A * B) is None
    assert (u * B).matches(x * A) is None
    assert (u * A * B).matches(x * A * B) == {u: x}
    assert (u * A * B).matches(x * B * A) is None
    assert (u * A * B).matches(x * A) is None
    assert (u * w * A).matches(x * A * B) is None
    assert (u * w * B).matches(x * A * B) == {u: x, w: A}
    assert (u * v * A * B).matches(x * A * B) in [{u: x, v: 1}, {v: x, u: 1}]
    assert (u * v * A * B).matches(x * B * A) is None
    assert (u * v * A * B).matches(u * v * A * C) is None