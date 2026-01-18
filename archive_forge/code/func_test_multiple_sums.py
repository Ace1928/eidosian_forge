from sympy.concrete.summations import Sum
from sympy.core.expr import Expr
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises
from sympy.printing.tensorflow import TensorflowPrinter
from sympy.printing.lambdarepr import lambdarepr, LambdaPrinter, NumExprPrinter
def test_multiple_sums():
    s = Sum(i * x + j, (i, a, b), (j, c, d))
    l = lambdarepr(s)
    assert l == '(builtins.sum(i*x + j for i in range(a, b+1) for j in range(c, d+1)))'
    args = (x, a, b, c, d)
    f = lambdify(args, s)
    vals = (2, 3, 4, 5, 6)
    f_ref = s.subs(zip(args, vals)).doit()
    f_res = f(*vals)
    assert f_res == f_ref