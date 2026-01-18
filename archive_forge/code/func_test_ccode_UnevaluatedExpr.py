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
def test_ccode_UnevaluatedExpr():
    assert ccode(UnevaluatedExpr(y * x) + z) == 'z + x*y'
    assert ccode(UnevaluatedExpr(y + x) + z) == 'z + (x + y)'
    w = symbols('w')
    assert ccode(UnevaluatedExpr(y + x) + UnevaluatedExpr(z + w)) == '(w + z) + (x + y)'
    p, q, r = symbols('p q r', real=True)
    q_r = UnevaluatedExpr(q + r)
    expr = abs(exp(p + q_r))
    assert ccode(expr) == 'exp(p + (q + r))'