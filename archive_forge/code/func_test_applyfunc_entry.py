from sympy.core.symbol import symbols, Dummy
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.core.function import Lambda
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.matmul import MatMul
from sympy.simplify.simplify import simplify
def test_applyfunc_entry():
    af = X.applyfunc(sin)
    assert af[0, 0] == sin(X[0, 0])
    af = Xd.applyfunc(sin)
    assert af[0, 0] == sin(X[0, 0])