from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.functions import express, matrix_to_vector, orthogonalize
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.testing.pytest import raises
def test_matrix_to_vector():
    m = Matrix([[1], [2], [3]])
    assert matrix_to_vector(m, C) == C.i + 2 * C.j + 3 * C.k
    m = Matrix([[0], [0], [0]])
    assert matrix_to_vector(m, N) == matrix_to_vector(m, C) == Vector.zero
    m = Matrix([[q1], [q2], [q3]])
    assert matrix_to_vector(m, N) == q1 * N.i + q2 * N.j + q3 * N.k