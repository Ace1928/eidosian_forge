from sympy.core import (S, pi, oo, Symbol, symbols, Rational, Integer,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.sets import Range
from sympy.logic import ITE
from sympy.codegen import For, aug_assign, Assignment
from sympy.testing.pytest import raises
from sympy.printing.rcode import RCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.rcode import rcode
def test_Matrix_printing():
    mat = Matrix([x * y, Piecewise((2 + x, y > 0), (y, True)), sin(z)])
    A = MatrixSymbol('A', 3, 1)
    p = rcode(mat, A)
    assert p == 'A[0] = x*y;\nA[1] = ifelse(y > 0,x + 2,y);\nA[2] = sin(z);'
    expr = Piecewise((2 * A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    p = rcode(expr)
    assert p == 'ifelse(x > 0,2*A[2],A[2]) + sin(A[1]) + A[0]'
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1, 0]), 0, cos(q[2, 0])], [q[1, 0] + q[2, 0], q[3, 0], 5], [2 * q[4, 0] / q[1, 0], sqrt(q[0, 0]) + 4, 0]])
    assert rcode(m, M) == 'M[0] = sin(q[1]);\nM[1] = 0;\nM[2] = cos(q[2]);\nM[3] = q[1] + q[2];\nM[4] = q[3];\nM[5] = 5;\nM[6] = 2*q[4]/q[1];\nM[7] = sqrt(q[0]) + 4;\nM[8] = 0;'