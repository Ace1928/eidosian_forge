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
def test_get_diag_blocks1():
    a = Matrix([[1, 2], [2, 3]])
    b = Matrix([[3, x], [y, 3]])
    c = Matrix([[3, x, 3], [y, 3, z], [x, y, z]])
    assert a.get_diag_blocks() == [a]
    assert b.get_diag_blocks() == [b]
    assert c.get_diag_blocks() == [c]