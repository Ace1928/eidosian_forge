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
def test_ccode_Relational():
    assert ccode(Eq(x, y)) == 'x == y'
    assert ccode(Ne(x, y)) == 'x != y'
    assert ccode(Le(x, y)) == 'x <= y'
    assert ccode(Lt(x, y)) == 'x < y'
    assert ccode(Gt(x, y)) == 'x > y'
    assert ccode(Ge(x, y)) == 'x >= y'