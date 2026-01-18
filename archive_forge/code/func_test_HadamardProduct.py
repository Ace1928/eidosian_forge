from sympy.matrices.dense import Matrix, eye
from sympy.matrices.common import ShapeError
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.special import Identity, OneMatrix, ZeroMatrix
from sympy.core import symbols
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.matrices import MatrixSymbol
from sympy.matrices.expressions import (HadamardProduct, hadamard_product, HadamardPower, hadamard_power)
def test_HadamardProduct():
    assert HadamardProduct(A, B, A).shape == A.shape
    raises(TypeError, lambda: HadamardProduct(A, n))
    raises(TypeError, lambda: HadamardProduct(A, 1))
    assert HadamardProduct(A, 2 * B, -A)[1, 1] == -2 * A[1, 1] * B[1, 1] * A[1, 1]
    mix = HadamardProduct(Z * A, B) * C
    assert mix.shape == (n, k)
    assert set(HadamardProduct(A, B, A).T.args) == {A.T, A.T, B.T}