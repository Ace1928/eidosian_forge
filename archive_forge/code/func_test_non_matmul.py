from sympy.assumptions import Q
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.numbers import I, Integer, oo, pi, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.common import (ShapeError, NonSquareMatrixError,
from sympy.matrices.matrices import MatrixCalculus
from sympy.matrices import (Matrix, diag, eye,
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import flatten
from sympy.testing.pytest import raises, XFAIL
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray as Array
from sympy.abc import x, y, z
def test_non_matmul():
    """
    Test that if explicitly specified as non-matrix, mul reverts
    to scalar multiplication.
    """

    class foo(Expr):
        is_Matrix = False
        is_MatrixLike = False
        shape = (1, 1)
    A = Matrix([[1, 2], [3, 4]])
    b = foo()
    assert b * A == Matrix([[b, 2 * b], [3 * b, 4 * b]])
    assert A * b == Matrix([[b, 2 * b], [3 * b, 4 * b]])