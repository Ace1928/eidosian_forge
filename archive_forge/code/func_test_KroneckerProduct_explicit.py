from sympy.core.mod import Mod
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import (Matrix, eye)
from sympy.matrices import MatrixSymbol, Identity
from sympy.matrices.expressions import det, trace
from sympy.matrices.expressions.kronecker import (KroneckerProduct,
def test_KroneckerProduct_explicit():
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    kp = KroneckerProduct(X, Y)
    assert kp.shape == (4, 4)
    assert kp.as_explicit() == Matrix([[X[0, 0] * Y[0, 0], X[0, 0] * Y[0, 1], X[0, 1] * Y[0, 0], X[0, 1] * Y[0, 1]], [X[0, 0] * Y[1, 0], X[0, 0] * Y[1, 1], X[0, 1] * Y[1, 0], X[0, 1] * Y[1, 1]], [X[1, 0] * Y[0, 0], X[1, 0] * Y[0, 1], X[1, 1] * Y[0, 0], X[1, 1] * Y[0, 1]], [X[1, 0] * Y[1, 0], X[1, 0] * Y[1, 1], X[1, 1] * Y[1, 0], X[1, 1] * Y[1, 1]]])