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
def test_is_anti_symmetric():
    x = symbols('x')
    assert PropertiesOnlyMatrix(2, 1, [1, 2]).is_anti_symmetric() is False
    m = PropertiesOnlyMatrix(3, 3, [0, x ** 2 + 2 * x + 1, y, -(x + 1) ** 2, 0, x * y, -y, -x * y, 0])
    assert m.is_anti_symmetric() is True
    assert m.is_anti_symmetric(simplify=False) is False
    assert m.is_anti_symmetric(simplify=lambda x: x) is False
    m = PropertiesOnlyMatrix(3, 3, [x.expand() for x in m])
    assert m.is_anti_symmetric(simplify=False) is True
    m = PropertiesOnlyMatrix(3, 3, [x.expand() for x in [S.One] + list(m)[1:]])
    assert m.is_anti_symmetric() is False