from sympy import sin, cos
from sympy.testing.pytest import raises
from sympy.tensor.toperators import PartialDerivative
from sympy.tensor.tensor import (TensorIndexType,
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.matrices.dense import diag
from sympy.tensor.array import Array
from sympy.core.random import randint
def test_eval_partial_derivative_expr1():
    tau, alpha = symbols('tau alpha')
    base_expr1 = A(i) * H(-i, j) + A(i) * A(-i) * A(j) + tau ** alpha * A(j)
    tensor_derivative = PartialDerivative(base_expr1, H(k, m))._perform_derivative()
    vector_derivative = PartialDerivative(base_expr1, A(k))._perform_derivative()
    scalar_derivative = PartialDerivative(base_expr1, tau)._perform_derivative()
    assert tensor_derivative - A(L_0) * L.metric(-L_0, -L_1) * L.delta(L_1, -k) * L.delta(j, -m) == 0
    assert (vector_derivative - (tau ** alpha * L.delta(j, -k) + L.delta(L_0, -k) * A(-L_0) * A(j) + A(L_0) * L.metric(-L_0, -L_1) * L.delta(L_1, -k) * A(j) + A(L_0) * A(-L_0) * L.delta(j, -k) + L.delta(L_0, -k) * H(-L_0, j))).expand() == 0
    assert (vector_derivative.contract_metric(L.metric).contract_delta(L.delta) - (tau ** alpha * L.delta(j, -k) + A(L_0) * A(-L_0) * L.delta(j, -k) + H(-k, j) + 2 * A(j) * A(-k))).expand() == 0
    assert scalar_derivative - alpha * 1 / tau * tau ** alpha * A(j) == 0