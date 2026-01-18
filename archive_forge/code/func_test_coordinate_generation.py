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
def test_coordinate_generation():
    q, u, qj, uj = dynamicsymbols('q u q_J u_J')
    q0j, q1j, q2j, q3j, u0j, u1j, u2j, u3j = dynamicsymbols('q0:4_J u0:4_J')
    q0, q1, q2, q3, u0, u1, u2, u3 = dynamicsymbols('q0:4 u0:4')
    _, _, P, C = _generate_body()
    J = PinJoint('J', P, C)
    assert J._fill_coordinate_list(q, 1) == Matrix([q])
    assert J._fill_coordinate_list([u], 1) == Matrix([u])
    assert J._fill_coordinate_list([u], 1, offset=2) == Matrix([u])
    assert J._fill_coordinate_list(None, 1) == Matrix([qj])
    assert J._fill_coordinate_list([None], 1) == Matrix([qj])
    assert J._fill_coordinate_list([q0, None, None], 3) == Matrix([q0, q1j, q2j])
    assert J._fill_coordinate_list(None, 3) == Matrix([q0j, q1j, q2j])
    assert J._fill_coordinate_list([], 3) == Matrix([q0j, q1j, q2j])
    assert J._fill_coordinate_list([], 3, offset=1) == Matrix([q1j, q2j, q3j])
    assert J._fill_coordinate_list([q1, None, q3], 3, offset=1) == Matrix([q1, q2j, q3])
    assert J._fill_coordinate_list(None, 2, offset=2) == Matrix([q2j, q3j])
    assert J._fill_coordinate_list(None, 1, 'u') == Matrix([uj])
    assert J._fill_coordinate_list([], 3, 'u') == Matrix([u0j, u1j, u2j])
    assert J._fill_coordinate_list(None, 1, number_single=True) == Matrix([q0j])
    assert J._fill_coordinate_list([], 1, 'u', 2, True) == Matrix([u2j])
    assert J._fill_coordinate_list([], 3, 'q') == Matrix([q0j, q1j, q2j])
    raises(ValueError, lambda: J._fill_coordinate_list([q0, q1], 1))
    raises(ValueError, lambda: J._fill_coordinate_list([u0, u1, None], 2, 'u'))
    raises(ValueError, lambda: J._fill_coordinate_list([q0, q1], 3))
    raises(TypeError, lambda: J._fill_coordinate_list([q0, symbols('q1')], 2))
    raises(TypeError, lambda: J._fill_coordinate_list([q0 + q1, q1], 2))
    _, _, P, C = _generate_body()
    PinJoint('J', P, C, q1, q1.diff(t))
    _, _, P, C = _generate_body()
    raises(ValueError, lambda: SphericalJoint('J', P, C, [q1j, None, None]))
    raises(ValueError, lambda: SphericalJoint('J', P, C, speeds=[u0, u0, u1]))