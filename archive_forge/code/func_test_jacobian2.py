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
def test_jacobian2():
    rho, phi = symbols('rho,phi')
    X = CalculusOnlyMatrix(3, 1, [rho * cos(phi), rho * sin(phi), rho ** 2])
    Y = CalculusOnlyMatrix(2, 1, [rho, phi])
    J = Matrix([[cos(phi), -rho * sin(phi)], [sin(phi), rho * cos(phi)], [2 * rho, 0]])
    assert X.jacobian(Y) == J
    m = CalculusOnlyMatrix(2, 2, [1, 2, 3, 4])
    m2 = CalculusOnlyMatrix(4, 1, [1, 2, 3, 4])
    raises(TypeError, lambda: m.jacobian(Matrix([1, 2])))
    raises(TypeError, lambda: m2.jacobian(m))