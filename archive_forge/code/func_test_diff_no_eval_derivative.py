from sympy.concrete.summations import Sum
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, diff, Subs)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.tensor.array.ndim_array import NDimArray
from sympy.testing.pytest import raises
from sympy.abc import a, b, c, x, y, z
def test_diff_no_eval_derivative():

    class My(Expr):

        def __new__(cls, x):
            return Expr.__new__(cls, x)
    assert My(x).diff(x).func is Derivative
    assert My(x).diff(x, 3).func is Derivative
    assert re(x).diff(x, 2) == Derivative(re(x), (x, 2))
    assert diff(NDimArray([re(x), im(x)]), (x, 2)) == NDimArray([Derivative(re(x), (x, 2)), Derivative(im(x), (x, 2))])
    assert My(x).diff(y) == 0