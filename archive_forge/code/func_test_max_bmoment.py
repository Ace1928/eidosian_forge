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
def test_max_bmoment():
    E = Symbol('E')
    I = Symbol('I')
    l, P = symbols('l, P', positive=True)
    b = Beam(l, E, I)
    R1, R2 = symbols('R1, R2')
    b.apply_load(R1, 0, -1)
    b.apply_load(R2, l, -1)
    b.apply_load(P, l / 2, -1)
    b.solve_for_reaction_loads(R1, R2)
    b.reaction_loads
    assert b.max_bmoment() == (l / 2, P * l / 4)
    b = Beam(l, E, I)
    R1, R2 = symbols('R1, R2')
    b.apply_load(R1, 0, -1)
    b.apply_load(R2, l, -1)
    b.apply_load(P, 0, 0, end=l)
    b.solve_for_reaction_loads(R1, R2)
    assert b.max_bmoment() == (l / 2, P * l ** 2 / 8)