from sympy import sin, cos
from sympy.testing.pytest import raises
from sympy.tensor.toperators import PartialDerivative
from sympy.tensor.tensor import (TensorIndexType,
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.matrices.dense import diag
from sympy.tensor.array import Array
from sympy.core.random import randint
def test_expand_partial_derivative_product_rule():
    expr4a = PartialDerivative(A(i) * B(j), D(k))
    assert expr4a._expand_partial_derivative() == PartialDerivative(A(i), D(k)) * B(j) + A(i) * PartialDerivative(B(j), D(k))
    expr4b = PartialDerivative(A(i) * B(j) * C(k), D(m))
    assert expr4b._expand_partial_derivative() == PartialDerivative(A(i), D(m)) * B(j) * C(k) + A(i) * PartialDerivative(B(j), D(m)) * C(k) + A(i) * B(j) * PartialDerivative(C(k), D(m))
    expr4c = PartialDerivative(A(i) * B(j), C(k), D(m))
    assert expr4c._expand_partial_derivative() == PartialDerivative(A(i), C(k), D(m)) * B(j) + PartialDerivative(A(i), C(k)) * PartialDerivative(B(j), D(m)) + PartialDerivative(A(i), D(m)) * PartialDerivative(B(j), C(k)) + A(i) * PartialDerivative(B(j), C(k), D(m))