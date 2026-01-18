from sympy.core.symbol import symbols, Dummy
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.core.function import Lambda
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.matmul import MatMul
from sympy.simplify.simplify import simplify
def test_applyfunc_matrix():
    x = Dummy('x')
    double = Lambda(x, x ** 2)
    expr = ElementwiseApplyFunction(double, Xd)
    assert isinstance(expr, ElementwiseApplyFunction)
    assert expr.doit() == Xd.applyfunc(lambda x: x ** 2)
    assert expr.shape == (3, 3)
    assert expr.func(*expr.args) == expr
    assert simplify(expr) == expr
    assert expr[0, 0] == double(Xd[0, 0])
    expr = ElementwiseApplyFunction(double, X)
    assert isinstance(expr, ElementwiseApplyFunction)
    assert isinstance(expr.doit(), ElementwiseApplyFunction)
    assert expr == X.applyfunc(double)
    assert expr.func(*expr.args) == expr
    expr = ElementwiseApplyFunction(exp, X * Y)
    assert expr.expr == X * Y
    assert expr.function.dummy_eq(Lambda(x, exp(x)))
    assert expr.dummy_eq((X * Y).applyfunc(exp))
    assert expr.func(*expr.args) == expr
    assert isinstance(X * expr, MatMul)
    assert (X * expr).shape == (3, 3)
    Z = MatrixSymbol('Z', 2, 3)
    assert (Z * expr).shape == (2, 3)
    expr = ElementwiseApplyFunction(exp, Z.T) * ElementwiseApplyFunction(exp, Z)
    assert expr.shape == (3, 3)
    expr = ElementwiseApplyFunction(exp, Z) * ElementwiseApplyFunction(exp, Z.T)
    assert expr.shape == (2, 2)
    M = Matrix([[x, y], [z, t]])
    expr = ElementwiseApplyFunction(sin, M)
    assert isinstance(expr, ElementwiseApplyFunction)
    assert expr.function.dummy_eq(Lambda(x, sin(x)))
    assert expr.expr == M
    assert expr.doit() == M.applyfunc(sin)
    assert expr.doit() == Matrix([[sin(x), sin(y)], [sin(z), sin(t)]])
    assert expr.func(*expr.args) == expr
    expr = ElementwiseApplyFunction(double, Xk)
    assert expr.doit() == expr
    assert expr.subs(k, 2).shape == (2, 2)
    assert (expr * expr).shape == (k, k)
    M = MatrixSymbol('M', k, t)
    expr2 = M.T * expr * M
    assert isinstance(expr2, MatMul)
    assert expr2.args[1] == expr
    assert expr2.shape == (t, t)
    expr3 = expr * M
    assert expr3.shape == (k, t)
    expr1 = ElementwiseApplyFunction(lambda x: x + 1, Xk)
    expr2 = ElementwiseApplyFunction(lambda x: x, Xk)
    assert expr1 != expr2