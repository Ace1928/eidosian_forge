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
def test_Dimension_error_definition():
    raises(TypeError, lambda: Dimension(('length', 1, 2)))
    raises(TypeError, lambda: Dimension(['length']))
    raises(TypeError, lambda: Dimension({'length': 'a'}))
    raises(TypeError, lambda: Dimension({'length': (1, 2)}))
    raises(AssertionError, lambda: Dimension('length', symbol=1))