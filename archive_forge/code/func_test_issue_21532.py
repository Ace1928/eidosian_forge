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
def test_issue_21532():
    f = Function('f')
    g = Function('g')
    FUNC_F, FUNC_G = symbols('FUNC_F, FUNC_G')
    assert f(x).count_ops(visual=True) == FUNC_F
    assert g(x).count_ops(visual=True) == FUNC_G