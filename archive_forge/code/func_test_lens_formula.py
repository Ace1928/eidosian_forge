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
def test_lens_formula():
    u, v, f = symbols('u, v, f')
    assert lens_formula(focal_length=f, u=u) == f * u / (f + u)
    assert lens_formula(focal_length=f, v=v) == f * v / (f - v)
    assert lens_formula(u=u, v=v) == u * v / (u - v)
    assert lens_formula(u=oo, v=v) == v
    assert lens_formula(u=oo, v=oo) is oo
    assert lens_formula(focal_length=oo, u=u) == u
    assert lens_formula(u=u, v=oo) == -u
    assert lens_formula(focal_length=oo, v=oo) is -oo
    assert lens_formula(focal_length=oo, v=v) == v
    assert lens_formula(focal_length=f, v=oo) == -f
    assert lens_formula(focal_length=oo, u=oo) is oo
    assert lens_formula(focal_length=oo, u=u) == u
    assert lens_formula(focal_length=f, u=oo) == f
    raises(ValueError, lambda: lens_formula(focal_length=f, u=u, v=v))