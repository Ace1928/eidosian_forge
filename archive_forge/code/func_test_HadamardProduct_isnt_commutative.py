from sympy.matrices.dense import Matrix, eye
from sympy.matrices.common import ShapeError
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.special import Identity, OneMatrix, ZeroMatrix
from sympy.core import symbols
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.matrices import MatrixSymbol
from sympy.matrices.expressions import (HadamardProduct, hadamard_product, HadamardPower, hadamard_power)
def test_HadamardProduct_isnt_commutative():
    assert HadamardProduct(A, B) != HadamardProduct(B, A)