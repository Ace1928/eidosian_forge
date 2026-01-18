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
def test_jordan_block():
    assert SpecialOnlyMatrix.jordan_block(3, 2) == SpecialOnlyMatrix.jordan_block(3, eigenvalue=2) == SpecialOnlyMatrix.jordan_block(size=3, eigenvalue=2) == SpecialOnlyMatrix.jordan_block(3, 2, band='upper') == SpecialOnlyMatrix.jordan_block(size=3, eigenval=2, eigenvalue=2) == Matrix([[2, 1, 0], [0, 2, 1], [0, 0, 2]])
    assert SpecialOnlyMatrix.jordan_block(3, 2, band='lower') == Matrix([[2, 0, 0], [1, 2, 0], [0, 1, 2]])
    raises(ValueError, lambda: SpecialOnlyMatrix.jordan_block(2))
    raises(ValueError, lambda: SpecialOnlyMatrix.jordan_block(3.5, 2))
    raises(ValueError, lambda: SpecialOnlyMatrix.jordan_block(eigenvalue=2))
    raises(ValueError, lambda: SpecialOnlyMatrix.jordan_block(eigenvalue=2, eigenval=4))
    assert SpecialOnlyMatrix.jordan_block(size=3, eigenvalue=2) == SpecialOnlyMatrix.jordan_block(size=3, eigenval=2)