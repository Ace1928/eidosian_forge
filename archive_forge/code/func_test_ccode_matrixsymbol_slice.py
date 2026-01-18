from io import StringIO
from sympy.core import symbols, Eq, pi, Catalan, Lambda, Dummy
from sympy.core.relational import Equality
from sympy.core.symbol import Symbol
from sympy.functions.special.error_functions import erf
from sympy.integrals.integrals import Integral
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import (
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
def test_ccode_matrixsymbol_slice():
    A = MatrixSymbol('A', 5, 3)
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    D = MatrixSymbol('D', 5, 1)
    name_expr = ('test', [Equality(B, A[0, :]), Equality(C, A[1, :]), Equality(D, A[:, 2])])
    result = codegen(name_expr, 'c99', 'test', header=False, empty=False)
    source = result[0][1]
    expected = '#include "test.h"\n#include <math.h>\nvoid test(double *A, double *B, double *C, double *D) {\n   B[0] = A[0];\n   B[1] = A[1];\n   B[2] = A[2];\n   C[0] = A[3];\n   C[1] = A[4];\n   C[2] = A[5];\n   D[0] = A[2];\n   D[1] = A[5];\n   D[2] = A[8];\n   D[3] = A[11];\n   D[4] = A[14];\n}\n'
    assert source == expected