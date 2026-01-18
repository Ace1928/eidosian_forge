from sympy.core.mod import Mod
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import (Matrix, eye)
from sympy.matrices import MatrixSymbol, Identity
from sympy.matrices.expressions import det, trace
from sympy.matrices.expressions.kronecker import (KroneckerProduct,
def test_KroneckerProduct_combine_pow():
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)
    assert combine_kronecker(KroneckerProduct(X, Y) ** x) == KroneckerProduct(X ** x, Y ** x)
    assert combine_kronecker(x * KroneckerProduct(X, Y) ** 2) == x * KroneckerProduct(X ** 2, Y ** 2)
    assert combine_kronecker(x * KroneckerProduct(X, Y) ** 2 * KroneckerProduct(A, B)) == x * KroneckerProduct(X ** 2 * A, Y ** 2 * B)
    assert combine_kronecker(KroneckerProduct(A, B.T) ** m) == KroneckerProduct(A, B.T) ** m