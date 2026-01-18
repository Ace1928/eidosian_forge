from sympy.core.mod import Mod
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import (Matrix, eye)
from sympy.matrices import MatrixSymbol, Identity
from sympy.matrices.expressions import det, trace
from sympy.matrices.expressions.kronecker import (KroneckerProduct,
def test_KroneckerProduct_entry():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', o, p)
    assert KroneckerProduct(A, B)._entry(i, j) == A[Mod(floor(i / o), n), Mod(floor(j / p), m)] * B[Mod(i, o), Mod(j, p)]