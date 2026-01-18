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
def test_convert_to():
    q = Quantity('q1')
    q.set_global_relative_scale_factor(S(5000), meter)
    assert q.convert_to(m) == 5000 * m
    assert speed_of_light.convert_to(m / s) == 299792458 * m / s
    assert day.convert_to(s) == 86400 * s
    assert q.convert_to(s) == q
    assert speed_of_light.convert_to(m) == speed_of_light
    expr = joule * second
    conv = convert_to(expr, joule)
    assert conv == joule * second