from sympy.concrete.summations import Sum
from sympy.core.exprtools import gcd_terms
from sympy.core.function import (diff, expand)
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol, Str)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.polys.polytools import factor
from sympy.core import (S, symbols, Add, Mul, SympifyError, Rational,
from sympy.functions import sin, cos, tan, sqrt, cbrt, exp
from sympy.simplify import simplify
from sympy.matrices import (ImmutableMatrix, Inverse, MatAdd, MatMul,
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.expressions.determinant import Determinant, det
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.testing.pytest import raises, XFAIL
def test_matexpr_indexing():
    A = MatrixSymbol('A', n, m)
    A[1, 2]
    A[l, k]
    A[l + 1, k + 1]
    A = MatrixSymbol('A', 2, 1)
    for i in range(-2, 2):
        for j in range(-1, 1):
            A[i, j]