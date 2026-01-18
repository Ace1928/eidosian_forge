from sympy.core.numbers import comp, Rational
from sympy.physics.optics.utils import (refraction_angle, fresnel_coefficients,
from sympy.physics.optics.medium import Medium
from sympy.physics.units import e0
from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.geometry.point import Point3D
from sympy.geometry.line import Ray3D
from sympy.geometry.plane import Plane
from sympy.testing.pytest import raises
def test_deviation():
    n1, n2 = symbols('n1, n2')
    r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))
    n = Matrix([0, 0, 1])
    i = Matrix([-1, -1, -1])
    normal_ray = Ray3D(Point3D(0, 0, 0), Point3D(0, 0, 1))
    P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])
    assert deviation(r1, 1, 1, normal=n) == 0
    assert deviation(r1, 1, 1, plane=P) == 0
    assert deviation(r1, 1, 1.1, plane=P).evalf(3) + 0.119 < 0.001
    assert deviation(i, 1, 1.1, normal=normal_ray).evalf(3) + 0.119 < 0.001
    assert deviation(r1, 1.33, 1, plane=P) is None
    assert deviation(r1, 1, 1, normal=[0, 0, 1]) == 0
    assert deviation([-1, -1, -1], 1, 1, normal=[0, 0, 1]) == 0
    assert ae(deviation(0.5, 1, 2), -0.25793, 5)
    assert ae(deviation(0.5, 2, 1), 0.78293, 5)