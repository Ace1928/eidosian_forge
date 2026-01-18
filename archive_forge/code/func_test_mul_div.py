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
def test_mul_div():
    u = Quantity('u')
    v = Quantity('v')
    t = Quantity('t')
    ut = Quantity('ut')
    v2 = Quantity('v')
    u.set_global_relative_scale_factor(S(10), meter)
    v.set_global_relative_scale_factor(S(5), meter)
    t.set_global_relative_scale_factor(S(2), second)
    ut.set_global_relative_scale_factor(S(20), meter * second)
    v2.set_global_relative_scale_factor(S(5), meter / second)
    assert 1 / u == u ** (-1)
    assert u / 1 == u
    v1 = u / t
    v2 = v
    assert v1 != v2
    assert v1 == v2.convert_to(v1)
    assert u * 1 == u
    ut1 = u * t
    ut2 = ut
    assert ut1 != ut2
    assert ut1 == ut2.convert_to(ut1)
    lp1 = Quantity('lp1')
    lp1.set_global_relative_scale_factor(S(2), 1 / meter)
    assert u * lp1 != 20
    assert u ** 0 == 1
    assert u ** 1 == u
    u2 = Quantity('u2')
    u3 = Quantity('u3')
    u2.set_global_relative_scale_factor(S(100), meter ** 2)
    u3.set_global_relative_scale_factor(Rational(1, 10), 1 / meter)
    assert u ** 2 != u2
    assert u ** (-1) != u3
    assert u ** 2 == u2.convert_to(u)
    assert u ** (-1) == u3.convert_to(u)