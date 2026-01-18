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
def test_as_real_imag():
    n = pi ** 1000
    assert n.as_real_imag() == (n, 0)
    x = Symbol('x')
    assert sqrt(x).as_real_imag() == ((re(x) ** 2 + im(x) ** 2) ** Rational(1, 4) * cos(atan2(im(x), re(x)) / 2), (re(x) ** 2 + im(x) ** 2) ** Rational(1, 4) * sin(atan2(im(x), re(x)) / 2))
    a, b = symbols('a,b', real=True)
    assert ((1 + sqrt(a + b * I)) / 2).as_real_imag() == ((a ** 2 + b ** 2) ** Rational(1, 4) * cos(atan2(b, a) / 2) / 2 + S.Half, (a ** 2 + b ** 2) ** Rational(1, 4) * sin(atan2(b, a) / 2) / 2)
    assert sqrt(a ** 2).as_real_imag() == (sqrt(a ** 2), 0)
    i = symbols('i', imaginary=True)
    assert sqrt(i ** 2).as_real_imag() == (0, abs(i))
    assert ((1 + I) / (1 - I)).as_real_imag() == (0, 1)
    assert ((1 + I) ** 3 / (1 - I)).as_real_imag() == (-2, 0)