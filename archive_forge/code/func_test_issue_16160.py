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
def test_issue_16160():
    assert Derivative(x ** 3, (x, x)).subs(x, 2) == Subs(Derivative(x ** 3, (x, 2)), x, 2)
    assert Derivative(1 + x ** 3, (x, x)).subs(x, 0) == Derivative(1 + y ** 3, (y, 0)).subs(y, 0)