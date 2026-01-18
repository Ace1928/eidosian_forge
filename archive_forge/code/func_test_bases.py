from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.physics.units import Quantity, length, meter
from sympy.physics.units.prefixes import PREFIXES, Prefix, prefix_unit, kilo, \
from sympy.physics.units.systems import SI
def test_bases():
    assert kilo.base == 10
    assert kibi.base == 2