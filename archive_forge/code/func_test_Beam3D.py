from sympy.core.function import expand
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.sets.sets import Interval
from sympy.simplify.simplify import simplify
from sympy.physics.continuum_mechanics.beam import Beam
from sympy.functions import SingularityFunction, Piecewise, meijerg, Abs, log
from sympy.testing.pytest import raises
from sympy.physics.units import meter, newton, kilo, giga, milli
from sympy.physics.continuum_mechanics.beam import Beam3D
from sympy.geometry import Circle, Polygon, Point2D, Triangle
from sympy.core.sympify import sympify
def test_Beam3D():
    l, E, G, I, A = symbols('l, E, G, I, A')
    R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
    b = Beam3D(l, E, G, I, A)
    m, q = symbols('m, q')
    b.apply_load(q, 0, 0, dir='y')
    b.apply_moment_load(m, 0, 0, dir='z')
    b.bc_slope = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    b.bc_deflection = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    b.solve_slope_deflection()
    assert b.polar_moment() == 2 * I
    assert b.shear_force() == [0, -q * x, 0]
    assert b.shear_stress() == [0, -q * x / A, 0]
    assert b.axial_stress() == 0
    assert b.bending_moment() == [0, 0, -m * x + q * x ** 2 / 2]
    expected_deflection = x * (A * G * q * x ** 3 / 4 + A * G * x ** 2 * (-l * (A * G * l * (l * q - 2 * m) + 12 * E * I * q) / (A * G * l ** 2 + 12 * E * I) / 2 - m) + 3 * E * I * l * (A * G * l * (l * q - 2 * m) + 12 * E * I * q) / (A * G * l ** 2 + 12 * E * I) + x * (-A * G * l ** 2 * q / 2 + 3 * A * G * l ** 2 * (A * G * l * (l * q - 2 * m) + 12 * E * I * q) / (A * G * l ** 2 + 12 * E * I) / 4 + A * G * l * m * Rational(3, 2) - 3 * E * I * q)) / (6 * A * E * G * I)
    dx, dy, dz = b.deflection()
    assert dx == dz == 0
    assert simplify(dy - expected_deflection) == 0
    b2 = Beam3D(30, E, G, I, A, x)
    b2.apply_load(50, start=0, order=0, dir='y')
    b2.bc_deflection = [(0, [0, 0, 0]), (30, [0, 0, 0])]
    b2.apply_load(R1, start=0, order=-1, dir='y')
    b2.apply_load(R2, start=30, order=-1, dir='y')
    b2.solve_for_reaction_loads(R1, R2)
    assert b2.reaction_loads == {R1: -750, R2: -750}
    b2.solve_slope_deflection()
    assert b2.slope() == [0, 0, 25 * x ** 3 / (3 * E * I) - 375 * x ** 2 / (E * I) + 3750 * x / (E * I)]
    expected_deflection = 25 * x ** 4 / (12 * E * I) - 125 * x ** 3 / (E * I) + 1875 * x ** 2 / (E * I) - 25 * x ** 2 / (A * G) + 750 * x / (A * G)
    dx, dy, dz = b2.deflection()
    assert dx == dz == 0
    assert dy == expected_deflection
    b3 = Beam3D(30, E, G, I, A, x)
    b3.apply_load(8, start=0, order=0, dir='y')
    b3.apply_load(9 * x, start=0, order=0, dir='z')
    b3.apply_load(R1, start=0, order=-1, dir='y')
    b3.apply_load(R2, start=30, order=-1, dir='y')
    b3.apply_load(R3, start=0, order=-1, dir='z')
    b3.apply_load(R4, start=30, order=-1, dir='z')
    b3.solve_for_reaction_loads(R1, R2, R3, R4)
    assert b3.reaction_loads == {R1: -120, R2: -120, R3: -1350, R4: -2700}