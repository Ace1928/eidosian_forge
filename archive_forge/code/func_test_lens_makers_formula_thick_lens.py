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
def test_lens_makers_formula_thick_lens():
    n1, n2 = symbols('n1, n2')
    m1 = Medium('m1', permittivity=e0, n=1)
    m2 = Medium('m2', permittivity=e0, n=1.33)
    assert ae(lens_makers_formula(m1, m2, 10, -10, d=1), -19.82, 2)
    assert lens_makers_formula(n1, n2, 1, -1, d=0.1) == n2 / ((2.0 - (0.1 * n1 - 0.1 * n2) / n1) * (n1 - n2))