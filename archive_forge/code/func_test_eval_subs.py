import warnings
from sympy.core.add import Add
from sympy.core.function import (Function, diff)
from sympy.core.numbers import (Number, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import integrate
from sympy.physics.units import (amount_of_substance, area, convert_to, find_unit,
from sympy.physics.units.definitions import (amu, au, centimeter, coulomb,
from sympy.physics.units.definitions.dimension_definitions import (
from sympy.physics.units.prefixes import PREFIXES, kilo
from sympy.physics.units.quantities import PhysicalConstant, Quantity
from sympy.physics.units.systems import SI
from sympy.testing.pytest import raises
def test_eval_subs():
    energy, mass, force = symbols('energy mass force')
    expr1 = energy / mass
    units = {energy: kilogram * meter ** 2 / second ** 2, mass: kilogram}
    assert expr1.subs(units) == meter ** 2 / second ** 2
    expr2 = force / mass
    units = {force: gravitational_constant * kilogram ** 2 / meter ** 2, mass: kilogram}
    assert expr2.subs(units) == gravitational_constant * kilogram / meter ** 2