from sympy.core import (
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.functions import (
from sympy.sets import Range
from sympy.logic import ITE, Implies, Equivalent
from sympy.codegen import For, aug_assign, Assignment
from sympy.testing.pytest import raises, XFAIL
from sympy.printing.c import C89CodePrinter, C99CodePrinter, get_math_macros
from sympy.codegen.ast import (
from sympy.codegen.cfunctions import expm1, log1p, exp2, log2, fma, log10, Cbrt, hypot, Sqrt
from sympy.codegen.cnodes import restrict
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol, SparseMatrix
from sympy.printing.codeprinter import ccode
def test_ccode_Rational():
    assert ccode(Rational(3, 7)) == '3.0/7.0'
    assert ccode(Rational(3, 7), type_aliases={real: float80}) == '3.0L/7.0L'
    assert ccode(Rational(18, 9)) == '2'
    assert ccode(Rational(3, -7)) == '-3.0/7.0'
    assert ccode(Rational(3, -7), type_aliases={real: float80}) == '-3.0L/7.0L'
    assert ccode(Rational(-3, -7)) == '3.0/7.0'
    assert ccode(Rational(-3, -7), type_aliases={real: float80}) == '3.0L/7.0L'
    assert ccode(x + Rational(3, 7)) == 'x + 3.0/7.0'
    assert ccode(x + Rational(3, 7), type_aliases={real: float80}) == 'x + 3.0L/7.0L'
    assert ccode(Rational(3, 7) * x) == '(3.0/7.0)*x'
    assert ccode(Rational(3, 7) * x, type_aliases={real: float80}) == '(3.0L/7.0L)*x'