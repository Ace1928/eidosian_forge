from sympy.core.numbers import (E, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core import Add, Mul, Pow
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x
def test_comparison_AccumBounds():
    assert (B(1, 3) < 4) == S.true
    assert (B(1, 3) < -1) == S.false
    assert (B(1, 3) < 2).rel_op == '<'
    assert (B(1, 3) <= 2).rel_op == '<='
    assert (B(1, 3) > 4) == S.false
    assert (B(1, 3) > -1) == S.true
    assert (B(1, 3) > 2).rel_op == '>'
    assert (B(1, 3) >= 2).rel_op == '>='
    assert (B(1, 3) < B(4, 6)) == S.true
    assert (B(1, 3) < B(2, 4)).rel_op == '<'
    assert (B(1, 3) < B(-2, 0)) == S.false
    assert (B(1, 3) <= B(4, 6)) == S.true
    assert (B(1, 3) <= B(-2, 0)) == S.false
    assert (B(1, 3) > B(4, 6)) == S.false
    assert (B(1, 3) > B(-2, 0)) == S.true
    assert (B(1, 3) >= B(4, 6)) == S.false
    assert (B(1, 3) >= B(-2, 0)) == S.true
    assert (cos(x) > 0).subs(x, oo) == (B(-1, 1) > 0)
    c = Symbol('c')
    raises(TypeError, lambda: B(0, 1) < c)
    raises(TypeError, lambda: B(0, 1) <= c)
    raises(TypeError, lambda: B(0, 1) > c)
    raises(TypeError, lambda: B(0, 1) >= c)