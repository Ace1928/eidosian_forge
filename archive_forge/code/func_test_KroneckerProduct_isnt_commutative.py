from sympy.core.mod import Mod
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import (Matrix, eye)
from sympy.matrices import MatrixSymbol, Identity
from sympy.matrices.expressions import det, trace
from sympy.matrices.expressions.kronecker import (KroneckerProduct,
def test_KroneckerProduct_isnt_commutative():
    assert KroneckerProduct(A, B) != KroneckerProduct(B, A)
    assert KroneckerProduct(A, B).is_commutative is False