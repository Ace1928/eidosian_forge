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
def test_ccode_Piecewise_deep():
    p = ccode(2 * Piecewise((x, x < 1), (x + 1, x < 2), (x ** 2, True)))
    assert p == '2*((x < 1) ? (\n   x\n)\n: ((x < 2) ? (\n   x + 1\n)\n: (\n   pow(x, 2)\n)))'
    expr = x * y * z + x ** 2 + y ** 2 + Piecewise((0, x < 0.5), (1, True)) + cos(z) - 1
    assert ccode(expr) == 'pow(x, 2) + x*y*z + pow(y, 2) + ((x < 0.5) ? (\n   0\n)\n: (\n   1\n)) + cos(z) - 1'
    assert ccode(expr, assign_to='c') == 'c = pow(x, 2) + x*y*z + pow(y, 2) + ((x < 0.5) ? (\n   0\n)\n: (\n   1\n)) + cos(z) - 1;'