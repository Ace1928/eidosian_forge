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
def test_ccode_loops_multiple_contractions():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)
    s = 'for (int i=0; i<m; i++){\n   y[i] = 0;\n}\nfor (int i=0; i<m; i++){\n   for (int j=0; j<n; j++){\n      for (int k=0; k<o; k++){\n         for (int l=0; l<p; l++){\n            y[i] = a[%s]*b[%s] + y[i];\n' % (i * n * o * p + j * o * p + k * p + l, j * o * p + k * p + l) + '         }\n      }\n   }\n}'
    assert ccode(b[j, k, l] * a[i, j, k, l], assign_to=y[i]) == s