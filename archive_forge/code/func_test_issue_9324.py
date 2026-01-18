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
def test_issue_9324():

    def count(val):
        return count_ops(val, visual=False)
    M = MatrixSymbol('M', 10, 10)
    assert count(M[0, 0]) == 0
    assert count(2 * M[0, 0] + M[5, 7]) == 2
    P = MatrixSymbol('P', 3, 3)
    Q = MatrixSymbol('Q', 3, 3)
    assert count(P + Q) == 1
    m = Symbol('m', integer=True)
    n = Symbol('n', integer=True)
    M = MatrixSymbol('M', m + n, m * m)
    assert count(M[0, 1]) == 2