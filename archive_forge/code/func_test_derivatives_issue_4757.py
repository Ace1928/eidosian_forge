from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, Lambda, expand)
from sympy.core.numbers import (E, I, Rational, comp, nan, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, adjoint, arg, conjugate, im, re, sign, transpose)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, atan, atan2, cos, sin)
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.funcmatrix import FunctionMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.immutable import (ImmutableMatrix, ImmutableSparseMatrix)
from sympy.matrices import SparseMatrix
from sympy.sets.sets import Interval
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow
def test_derivatives_issue_4757():
    x = Symbol('x', real=True)
    y = Symbol('y', imaginary=True)
    f = Function('f')
    assert re(f(x)).diff(x) == re(f(x).diff(x))
    assert im(f(x)).diff(x) == im(f(x).diff(x))
    assert re(f(y)).diff(y) == -I * im(f(y).diff(y))
    assert im(f(y)).diff(y) == -I * re(f(y).diff(y))
    assert Abs(f(x)).diff(x).subs(f(x), 1 + I * x).doit() == x / sqrt(1 + x ** 2)
    assert arg(f(x)).diff(x).subs(f(x), 1 + I * x ** 2).doit() == 2 * x / (1 + x ** 4)
    assert Abs(f(y)).diff(y).subs(f(y), 1 + y).doit() == -y / sqrt(1 - y ** 2)
    assert arg(f(y)).diff(y).subs(f(y), I + y ** 2).doit() == 2 * y / (1 + y ** 4)