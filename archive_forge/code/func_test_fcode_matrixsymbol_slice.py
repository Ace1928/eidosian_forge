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
def test_fcode_matrixsymbol_slice():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    D = MatrixSymbol('D', 2, 1)
    name_expr = ('test', [Equality(B, A[0, :]), Equality(C, A[1, :]), Equality(D, A[:, 2])])
    result = codegen(name_expr, 'f95', 'test', header=False, empty=False)
    source = result[0][1]
    expected = 'subroutine test(A, B, C, D)\nimplicit none\nREAL*8, intent(in), dimension(1:2, 1:3) :: A\nREAL*8, intent(out), dimension(1:1, 1:3) :: B\nREAL*8, intent(out), dimension(1:1, 1:3) :: C\nREAL*8, intent(out), dimension(1:2, 1:1) :: D\nB(1, 1) = A(1, 1)\nB(1, 2) = A(1, 2)\nB(1, 3) = A(1, 3)\nC(1, 1) = A(2, 1)\nC(1, 2) = A(2, 2)\nC(1, 3) = A(2, 3)\nD(1, 1) = A(1, 3)\nD(2, 1) = A(2, 3)\nend subroutine\n'
    assert source == expected