from sympy import sin, cos
from sympy.testing.pytest import raises
from sympy.tensor.toperators import PartialDerivative
from sympy.tensor.tensor import (TensorIndexType,
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.matrices.dense import diag
from sympy.tensor.array import Array
from sympy.core.random import randint
def test_expand_partial_derivative_constant_factor_rule():
    nneg = randint(0, 1000)
    pos = randint(1, 1000)
    neg = -randint(1, 1000)
    c1 = Rational(nneg, pos)
    c2 = Rational(neg, pos)
    c3 = Rational(nneg, neg)
    expr2a = PartialDerivative(nneg * A(i), D(j))
    assert expr2a._expand_partial_derivative() == nneg * PartialDerivative(A(i), D(j))
    expr2b = PartialDerivative(neg * A(i), D(j))
    assert expr2b._expand_partial_derivative() == neg * PartialDerivative(A(i), D(j))
    expr2ca = PartialDerivative(c1 * A(i), D(j))
    assert expr2ca._expand_partial_derivative() == c1 * PartialDerivative(A(i), D(j))
    expr2cb = PartialDerivative(c2 * A(i), D(j))
    assert expr2cb._expand_partial_derivative() == c2 * PartialDerivative(A(i), D(j))
    expr2cc = PartialDerivative(c3 * A(i), D(j))
    assert expr2cc._expand_partial_derivative() == c3 * PartialDerivative(A(i), D(j))