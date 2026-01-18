from sympy.testing.pytest import raises
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.scalar import BaseScalar
from sympy.core.function import expand
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)
from sympy.matrices.dense import zeros
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.vector.functions import express
from sympy.vector.point import Point
from sympy.vector.vector import Vector
from sympy.vector.orienters import (AxisOrienter, BodyOrienter,
def test_func_args():
    A = CoordSys3D('A')
    assert A.x.func(*A.x.args) == A.x
    expr = 3 * A.x + 4 * A.y
    assert expr.func(*expr.args) == expr
    assert A.i.func(*A.i.args) == A.i
    v = A.x * A.i + A.y * A.j + A.z * A.k
    assert v.func(*v.args) == v
    assert A.origin.func(*A.origin.args) == A.origin