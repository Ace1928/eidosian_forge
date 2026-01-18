from sympy import sin, cos
from sympy.testing.pytest import raises
from sympy.tensor.toperators import PartialDerivative
from sympy.tensor.tensor import (TensorIndexType,
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.matrices.dense import diag
from sympy.tensor.array import Array
from sympy.core.random import randint
def test_eval_partial_derivative_single_2nd_rank_tensors_by_tensor():
    expr1 = PartialDerivative(H(i, j), H(m, m1))
    assert expr1._perform_derivative() - L.delta(i, -m) * L.delta(j, -m1) == 0
    expr2 = PartialDerivative(H(i, j), H(-m, m1))
    assert expr2._perform_derivative() - L.metric(i, L_0) * L.delta(-L_0, m) * L.delta(j, -m1) == 0
    expr3 = PartialDerivative(H(i, j), H(m, -m1))
    assert expr3._perform_derivative() - L.delta(i, -m) * L.metric(j, L_0) * L.delta(-L_0, m1) == 0
    expr4 = PartialDerivative(H(i, j), H(-m, -m1))
    assert expr4._perform_derivative() - L.metric(i, L_0) * L.delta(-L_0, m) * L.metric(j, L_1) * L.delta(-L_1, m1) == 0