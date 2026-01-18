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
def test_Beam():
    E = Symbol('E')
    E_1 = Symbol('E_1')
    I = Symbol('I')
    I_1 = Symbol('I_1')
    A = Symbol('A')
    b = Beam(1, E, I)
    assert b.length == 1
    assert b.elastic_modulus == E
    assert b.second_moment == I
    assert b.variable == x
    b.length = 4
    assert b.length == 4
    b.elastic_modulus = E_1
    assert b.elastic_modulus == E_1
    b.second_moment = I_1
    assert b.second_moment is I_1
    b.variable = y
    assert b.variable is y
    b.bc_deflection = [(0, 2)]
    b.bc_slope = [(0, 1)]
    assert b.boundary_conditions == {'deflection': [(0, 2)], 'slope': [(0, 1)]}
    b.bc_slope.extend([(4, 3), (5, 0)])
    s_bcs = b.bc_slope
    assert s_bcs == [(0, 1), (4, 3), (5, 0)]
    b.bc_deflection.extend([(4, 3), (5, 0)])
    d_bcs = b.bc_deflection
    assert d_bcs == [(0, 2), (4, 3), (5, 0)]
    bcs_new = b.boundary_conditions
    assert bcs_new == {'deflection': [(0, 2), (4, 3), (5, 0)], 'slope': [(0, 1), (4, 3), (5, 0)]}
    b1 = Beam(30, E, I)
    b1.apply_load(-8, 0, -1)
    b1.apply_load(R1, 10, -1)
    b1.apply_load(R2, 30, -1)
    b1.apply_load(120, 30, -2)
    b1.bc_deflection = [(10, 0), (30, 0)]
    b1.solve_for_reaction_loads(R1, R2)
    p = b1.reaction_loads
    q = {R1: 6, R2: 2}
    assert p == q
    p = b1.load
    q = -8 * SingularityFunction(x, 0, -1) + 6 * SingularityFunction(x, 10, -1) + 120 * SingularityFunction(x, 30, -2) + 2 * SingularityFunction(x, 30, -1)
    assert p == q
    p = b1.shear_force()
    q = 8 * SingularityFunction(x, 0, 0) - 6 * SingularityFunction(x, 10, 0) - 120 * SingularityFunction(x, 30, -1) - 2 * SingularityFunction(x, 30, 0)
    assert p == q
    p = b1.shear_stress()
    q = (8 * SingularityFunction(x, 0, 0) - 6 * SingularityFunction(x, 10, 0) - 120 * SingularityFunction(x, 30, -1) - 2 * SingularityFunction(x, 30, 0)) / A
    assert p == q
    p = b1.bending_moment()
    q = 8 * SingularityFunction(x, 0, 1) - 6 * SingularityFunction(x, 10, 1) - 120 * SingularityFunction(x, 30, 0) - 2 * SingularityFunction(x, 30, 1)
    assert p == q
    p = b1.slope()
    q = -4 * SingularityFunction(x, 0, 2) + 3 * SingularityFunction(x, 10, 2) + 120 * SingularityFunction(x, 30, 1) + SingularityFunction(x, 30, 2) + Rational(4000, 3)
    assert p == q / (E * I)
    p = b1.deflection()
    q = x * Rational(4000, 3) - 4 * SingularityFunction(x, 0, 3) / 3 + SingularityFunction(x, 10, 3) + 60 * SingularityFunction(x, 30, 2) + SingularityFunction(x, 30, 3) / 3 - 12000
    assert p == q / (E * I)
    l = Symbol('l')
    w0 = Symbol('w0')
    w2 = Symbol('w2')
    a1 = Symbol('a1')
    c = Symbol('c')
    c1 = Symbol('c1')
    d = Symbol('d')
    e = Symbol('e')
    f = Symbol('f')
    b2 = Beam(l, E, I)
    b2.apply_load(w0, a1, 1)
    b2.apply_load(w2, c1, -1)
    b2.bc_deflection = [(c, d)]
    b2.bc_slope = [(e, f)]
    p = b2.load
    q = w0 * SingularityFunction(x, a1, 1) + w2 * SingularityFunction(x, c1, -1)
    assert p == q
    p = b2.shear_force()
    q = -w0 * SingularityFunction(x, a1, 2) / 2 - w2 * SingularityFunction(x, c1, 0)
    assert p == q
    p = b2.shear_stress()
    q = (-w0 * SingularityFunction(x, a1, 2) / 2 - w2 * SingularityFunction(x, c1, 0)) / A
    assert p == q
    p = b2.bending_moment()
    q = -w0 * SingularityFunction(x, a1, 3) / 6 - w2 * SingularityFunction(x, c1, 1)
    assert p == q
    p = b2.slope()
    q = (w0 * SingularityFunction(x, a1, 4) / 24 + w2 * SingularityFunction(x, c1, 2) / 2) / (E * I) + (E * I * f - w0 * SingularityFunction(e, a1, 4) / 24 - w2 * SingularityFunction(e, c1, 2) / 2) / (E * I)
    assert expand(p) == expand(q)
    p = b2.deflection()
    q = x * (E * I * f - w0 * SingularityFunction(e, a1, 4) / 24 - w2 * SingularityFunction(e, c1, 2) / 2) / (E * I) + (w0 * SingularityFunction(x, a1, 5) / 120 + w2 * SingularityFunction(x, c1, 3) / 6) / (E * I) + (E * I * (-c * f + d) + c * w0 * SingularityFunction(e, a1, 4) / 24 + c * w2 * SingularityFunction(e, c1, 2) / 2 - w0 * SingularityFunction(c, a1, 5) / 120 - w2 * SingularityFunction(c, c1, 3) / 6) / (E * I)
    assert simplify(p - q) == 0
    b3 = Beam(9, E, I, 2)
    b3.apply_load(value=-2, start=2, order=2, end=3)
    b3.bc_slope.append((0, 2))
    C3 = symbols('C3')
    C4 = symbols('C4')
    p = b3.load
    q = -2 * SingularityFunction(x, 2, 2) + 2 * SingularityFunction(x, 3, 0) + 4 * SingularityFunction(x, 3, 1) + 2 * SingularityFunction(x, 3, 2)
    assert p == q
    p = b3.shear_force()
    q = 2 * SingularityFunction(x, 2, 3) / 3 - 2 * SingularityFunction(x, 3, 1) - 2 * SingularityFunction(x, 3, 2) - 2 * SingularityFunction(x, 3, 3) / 3
    assert p == q
    p = b3.shear_stress()
    q = SingularityFunction(x, 2, 3) / 3 - 1 * SingularityFunction(x, 3, 1) - 1 * SingularityFunction(x, 3, 2) - 1 * SingularityFunction(x, 3, 3) / 3
    assert p == q
    p = b3.slope()
    q = 2 - (SingularityFunction(x, 2, 5) / 30 - SingularityFunction(x, 3, 3) / 3 - SingularityFunction(x, 3, 4) / 6 - SingularityFunction(x, 3, 5) / 30) / (E * I)
    assert p == q
    p = b3.deflection()
    q = 2 * x - (SingularityFunction(x, 2, 6) / 180 - SingularityFunction(x, 3, 4) / 12 - SingularityFunction(x, 3, 5) / 30 - SingularityFunction(x, 3, 6) / 180) / (E * I)
    assert p == q + C4
    b4 = Beam(4, E, I, 3)
    b4.apply_load(-3, 0, 0, end=3)
    p = b4.load
    q = -3 * SingularityFunction(x, 0, 0) + 3 * SingularityFunction(x, 3, 0)
    assert p == q
    p = b4.shear_force()
    q = 3 * SingularityFunction(x, 0, 1) - 3 * SingularityFunction(x, 3, 1)
    assert p == q
    p = b4.shear_stress()
    q = SingularityFunction(x, 0, 1) - SingularityFunction(x, 3, 1)
    assert p == q
    p = b4.slope()
    q = -3 * SingularityFunction(x, 0, 3) / 6 + 3 * SingularityFunction(x, 3, 3) / 6
    assert p == q / (E * I) + C3
    p = b4.deflection()
    q = -3 * SingularityFunction(x, 0, 4) / 24 + 3 * SingularityFunction(x, 3, 4) / 24
    assert p == q / (E * I) + C3 * x + C4
    raises(ValueError, lambda: b4.apply_load(-3, 0, -1, end=3))
    with raises(TypeError):
        b4.variable = 1