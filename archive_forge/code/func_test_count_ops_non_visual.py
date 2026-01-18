from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import (Derivative, Function, count_ops)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (Eq, Rel)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Equivalent, ITE, Implies, Nand,
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.core.containers import Tuple
def test_count_ops_non_visual():

    def count(val):
        return count_ops(val, visual=False)
    assert count(x) == 0
    assert count(x) is not S.Zero
    assert count(x + y) == 1
    assert count(x + y) is not S.One
    assert count(x + y * x + 2 * y) == 4
    assert count({x + y: x}) == 1
    assert count({x + y: S(2) + x}) is not S.One
    assert count(x < y) == 1
    assert count(Or(x, y)) == 1
    assert count(And(x, y)) == 1
    assert count(Not(x)) == 1
    assert count(Nor(x, y)) == 2
    assert count(Nand(x, y)) == 2
    assert count(Xor(x, y)) == 1
    assert count(Implies(x, y)) == 1
    assert count(Equivalent(x, y)) == 1
    assert count(ITE(x, y, z)) == 1
    assert count(ITE(True, x, y)) == 0