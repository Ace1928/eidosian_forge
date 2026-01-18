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
def test_insufficient_bconditions():
    L = symbols('L', positive=True)
    E, I, P, a3, a4 = symbols('E I P a3 a4')
    b = Beam(L, E, I, base_char='a')
    b.apply_load(R2, L, -1)
    b.apply_load(R1, 0, -1)
    b.apply_load(-P, L / 2, -1)
    b.solve_for_reaction_loads(R1, R2)
    p = b.slope()
    q = P * SingularityFunction(x, 0, 2) / 4 - P * SingularityFunction(x, L / 2, 2) / 2 + P * SingularityFunction(x, L, 2) / 4
    assert p == q / (E * I) + a3
    p = b.deflection()
    q = P * SingularityFunction(x, 0, 3) / 12 - P * SingularityFunction(x, L / 2, 3) / 6 + P * SingularityFunction(x, L, 3) / 12
    assert p == q / (E * I) + a3 * x + a4
    b.bc_deflection = [(0, 0)]
    p = b.deflection()
    q = a3 * x + P * SingularityFunction(x, 0, 3) / 12 - P * SingularityFunction(x, L / 2, 3) / 6 + P * SingularityFunction(x, L, 3) / 12
    assert p == q / (E * I)
    b.bc_deflection = [(0, 0), (L, 0)]
    p = b.deflection()
    q = -L ** 2 * P * x / 16 + P * SingularityFunction(x, 0, 3) / 12 - P * SingularityFunction(x, L / 2, 3) / 6 + P * SingularityFunction(x, L, 3) / 12
    assert p == q / (E * I)