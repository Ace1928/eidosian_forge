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
def test_Dimension_definition():
    assert dimsys_SI.get_dimensional_dependencies(length) == {length: 1}
    assert length.name == Symbol('length')
    assert length.symbol == Symbol('L')
    halflength = sqrt(length)
    assert dimsys_SI.get_dimensional_dependencies(halflength) == {length: S.Half}