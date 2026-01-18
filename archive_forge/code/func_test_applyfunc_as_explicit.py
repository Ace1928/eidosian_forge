from sympy.core.symbol import symbols, Dummy
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.core.function import Lambda
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.matmul import MatMul
from sympy.simplify.simplify import simplify
def test_applyfunc_as_explicit():
    af = X.applyfunc(sin)
    assert af.as_explicit() == Matrix([[sin(X[0, 0]), sin(X[0, 1]), sin(X[0, 2])], [sin(X[1, 0]), sin(X[1, 1]), sin(X[1, 2])], [sin(X[2, 0]), sin(X[2, 1]), sin(X[2, 2])]])