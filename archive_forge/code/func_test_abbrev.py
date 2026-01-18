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
def test_abbrev():
    u = Quantity('u')
    u.set_global_relative_scale_factor(S.One, meter)
    assert u.name == Symbol('u')
    assert u.abbrev == Symbol('u')
    u = Quantity('u', abbrev='om')
    u.set_global_relative_scale_factor(S(2), meter)
    assert u.name == Symbol('u')
    assert u.abbrev == Symbol('om')
    assert u.scale_factor == 2
    assert isinstance(u.scale_factor, Number)
    u = Quantity('u', abbrev='ikm')
    u.set_global_relative_scale_factor(3 * kilo, meter)
    assert u.abbrev == Symbol('ikm')
    assert u.scale_factor == 3000