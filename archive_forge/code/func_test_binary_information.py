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
def test_binary_information():
    assert convert_to(kibibyte, byte) == 1024 * byte
    assert convert_to(mebibyte, byte) == 1024 ** 2 * byte
    assert convert_to(gibibyte, byte) == 1024 ** 3 * byte
    assert convert_to(tebibyte, byte) == 1024 ** 4 * byte
    assert convert_to(pebibyte, byte) == 1024 ** 5 * byte
    assert convert_to(exbibyte, byte) == 1024 ** 6 * byte
    assert kibibyte.convert_to(bit) == 8 * 1024 * bit
    assert byte.convert_to(bit) == 8 * bit
    a = 10 * kibibyte * hour
    assert convert_to(a, byte) == 10240 * byte * hour
    assert convert_to(a, minute) == 600 * kibibyte * minute
    assert convert_to(a, [byte, minute]) == 614400 * byte * minute