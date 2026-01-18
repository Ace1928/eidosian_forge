from sympy.core.random import randint
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, ones, zeros)
from sympy.physics.quantum.matrixutils import (
from sympy.external import import_module
from sympy.testing.pytest import skip
def test_matrix_to_zero():
    assert matrix_to_zero(m) == m
    assert matrix_to_zero(Matrix([[0, 0], [0, 0]])) == Integer(0)