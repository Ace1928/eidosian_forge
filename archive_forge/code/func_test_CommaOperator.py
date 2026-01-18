from sympy.core.symbol import symbols
from sympy.printing.codeprinter import ccode
from sympy.codegen.ast import Declaration, Variable, float64, int64, String, CodeBlock
from sympy.codegen.cnodes import (
def test_CommaOperator():
    expr = CommaOperator(PreIncrement(x), 2 * x)
    assert ccode(expr) == '(++(x), 2*x)'
    assert expr.func(*expr.args) == expr