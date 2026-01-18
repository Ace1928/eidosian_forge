from sympy.matrices.dense import Matrix, eye
from sympy.matrices.common import ShapeError
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.special import Identity, OneMatrix, ZeroMatrix
from sympy.core import symbols
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.matrices import MatrixSymbol
from sympy.matrices.expressions import (HadamardProduct, hadamard_product, HadamardPower, hadamard_power)
def test_hadamard_product_with_explicit_mat():
    A = MatrixSymbol('A', 3, 3).as_explicit()
    B = MatrixSymbol('B', 3, 3).as_explicit()
    X = MatrixSymbol('X', 3, 3)
    expr = hadamard_product(A, B)
    ret = Matrix([i * j for i, j in zip(A, B)]).reshape(3, 3)
    assert expr == ret
    expr = hadamard_product(A, X, B)
    assert expr == HadamardProduct(ret, X)
    expr = hadamard_product(eye(3), A)
    assert expr == Matrix([[A[0, 0], 0, 0], [0, A[1, 1], 0], [0, 0, A[2, 2]]])
    expr = hadamard_product(eye(3), eye(3))
    assert expr == eye(3)