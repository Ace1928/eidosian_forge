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
def test_ccode_math_macros():
    assert ccode(z + exp(1)) == 'z + M_E'
    assert ccode(z + log2(exp(1))) == 'z + M_LOG2E'
    assert ccode(z + 1 / log(2)) == 'z + M_LOG2E'
    assert ccode(z + log(2)) == 'z + M_LN2'
    assert ccode(z + log(10)) == 'z + M_LN10'
    assert ccode(z + pi) == 'z + M_PI'
    assert ccode(z + pi / 2) == 'z + M_PI_2'
    assert ccode(z + pi / 4) == 'z + M_PI_4'
    assert ccode(z + 1 / pi) == 'z + M_1_PI'
    assert ccode(z + 2 / pi) == 'z + M_2_PI'
    assert ccode(z + 2 / sqrt(pi)) == 'z + M_2_SQRTPI'
    assert ccode(z + 2 / Sqrt(pi)) == 'z + M_2_SQRTPI'
    assert ccode(z + sqrt(2)) == 'z + M_SQRT2'
    assert ccode(z + Sqrt(2)) == 'z + M_SQRT2'
    assert ccode(z + 1 / sqrt(2)) == 'z + M_SQRT1_2'
    assert ccode(z + 1 / Sqrt(2)) == 'z + M_SQRT1_2'