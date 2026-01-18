from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.expr import Expr
from sympy.core.numbers import (I, Rational, nan, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, atan2)
from sympy.abc import w, x, y, z
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.expressions.matexpr import MatrixSymbol
def test_eval_refine():

    class MockExpr(Expr):

        def _eval_refine(self, assumptions):
            return True
    mock_obj = MockExpr()
    assert refine(mock_obj)