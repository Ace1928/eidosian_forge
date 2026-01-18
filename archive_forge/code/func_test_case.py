from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda, diff)
from sympy.core.mod import Mod
from sympy.core import (Catalan, EulerGamma, GoldenRatio)
from sympy.core.numbers import (E, Float, I, Integer, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (conjugate, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.sets.fancysets import Range
from sympy.codegen import For, Assignment, aug_assign
from sympy.codegen.ast import Declaration, Variable, float32, float64, \
from sympy.core.expr import UnevaluatedExpr
from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Or, Not, Equivalent, Xor
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.fortran import fcode, FCodePrinter
from sympy.tensor import IndexedBase, Idx
from sympy.tensor.array.expressions import ArraySymbol, ArrayElement
from sympy.utilities.lambdify import implemented_function
from sympy.testing.pytest import raises
def test_case():
    ob = FCodePrinter()
    x, x_, x__, y, X, X_, Y = symbols('x,x_,x__,y,X,X_,Y')
    assert fcode(exp(x_) + sin(x * y) + cos(X * Y)) == '      exp(x_) + sin(x*y) + cos(X__*Y_)'
    assert fcode(exp(x__) + 2 * x * Y * X_ ** Rational(7, 2)) == '      2*X_**(7.0d0/2.0d0)*Y*x + exp(x__)'
    assert fcode(exp(x_) + sin(x * y) + cos(X * Y), name_mangling=False) == '      exp(x_) + sin(x*y) + cos(X*Y)'
    assert fcode(x - cos(X), name_mangling=False) == '      x - cos(X)'
    assert ob.doprint(X * sin(x) + x_, assign_to='me') == '      me = X*sin(x_) + x__'
    assert ob.doprint(X * sin(x), assign_to='mu') == '      mu = X*sin(x_)'
    assert ob.doprint(x_, assign_to='ad') == '      ad = x__'
    n, m = symbols('n,m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    I = Idx('I', n)
    assert fcode(A[i, I] * x[I], assign_to=y[i], source_format='free') == 'do i = 1, m\n   y(i) = 0\nend do\ndo i = 1, m\n   do I_ = 1, n\n      y(i) = A(i, I_)*x(I_) + y(i)\n   end do\nend do'