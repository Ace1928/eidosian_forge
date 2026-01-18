from sympy.core.mod import Mod
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import (Matrix, eye)
from sympy.matrices import MatrixSymbol, Identity
from sympy.matrices.expressions import det, trace
from sympy.matrices.expressions.kronecker import (KroneckerProduct,
def test_KroneckerProduct_expand():
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)
    assert KroneckerProduct(X + Y, Y + Z).expand(kroneckerproduct=True) == KroneckerProduct(X, Y) + KroneckerProduct(X, Z) + KroneckerProduct(Y, Y) + KroneckerProduct(Y, Z)