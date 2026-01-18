from sympy.core.function import expand_mul
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core.backend import Matrix, _simplify_matrix, eye, zeros
from sympy.core.symbol import symbols
from sympy.physics.mechanics import (dynamicsymbols, Body, JointsMethod,
from sympy.physics.mechanics.joint import Joint
from sympy.physics.vector import Vector, ReferenceFrame, Point
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_create_aligned_frame_pi():
    N, A, P, C = _generate_body()
    f = Joint._create_aligned_interframe(P, -P.x, P.x)
    assert f.z == P.z
    f = Joint._create_aligned_interframe(P, -P.y, P.y)
    assert f.x == P.x
    f = Joint._create_aligned_interframe(P, -P.z, P.z)
    assert f.y == P.y
    f = Joint._create_aligned_interframe(P, -P.x - P.y, P.x + P.y)
    assert f.z == P.z
    f = Joint._create_aligned_interframe(P, -P.y - P.z, P.y + P.z)
    assert f.x == P.x
    f = Joint._create_aligned_interframe(P, -P.x - P.z, P.x + P.z)
    assert f.y == P.y
    f = Joint._create_aligned_interframe(P, -P.x - P.y - P.z, P.x + P.y + P.z)
    assert f.y - f.z == P.y - P.z