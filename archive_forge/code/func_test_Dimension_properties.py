from sympy.physics.units.systems.si import dimsys_SI
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos)
from sympy.physics.units.dimensions import Dimension
from sympy.physics.units.definitions.dimension_definitions import (
from sympy.physics.units import foot
from sympy.testing.pytest import raises
def test_Dimension_properties():
    assert dimsys_SI.is_dimensionless(length) is False
    assert dimsys_SI.is_dimensionless(length / length) is True
    assert dimsys_SI.is_dimensionless(Dimension('undefined')) is False
    assert length.has_integer_powers(dimsys_SI) is True
    assert (length ** (-1)).has_integer_powers(dimsys_SI) is True
    assert (length ** 1.5).has_integer_powers(dimsys_SI) is False