from sympy import sin, cos
from sympy.testing.pytest import raises
from sympy.tensor.toperators import PartialDerivative
from sympy.tensor.tensor import (TensorIndexType,
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.matrices.dense import diag
from sympy.tensor.array import Array
from sympy.core.random import randint
def test_eval_partial_derivative_mixed_scalar_tensor_expr2():
    tau, alpha = symbols('tau alpha')
    base_expr2 = A(i) * A(-i) + tau ** 2
    vector_expression = PartialDerivative(base_expr2, A(k))._perform_derivative()
    assert (vector_expression - (L.delta(L_0, -k) * A(-L_0) + A(L_0) * L.metric(-L_0, -L_1) * L.delta(L_1, -k))).expand() == 0
    scalar_expression = PartialDerivative(base_expr2, tau)._perform_derivative()
    assert scalar_expression == 2 * tau