from functools import reduce
from operator import attrgetter, add
import sys
from sympy import nsimplify
import pytest
from ..util.arithmeticdict import ArithmeticDict
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, to_unitless, allclose
from ..chemistry import (
@requires(parsing_library, units_library)
def test_Equilibrium__from_string():
    assert Equilibrium.from_string('H2O = H+ + OH-').param is None
    assert Equilibrium.from_string('H2O = H+ + OH-; 1e-14').param == 1e-14
    assert Equilibrium.from_string('H2O = H+ + OH-; 1e-14*molar').param ** 0 == 1
    with pytest.raises(ValueError):
        Equilibrium.from_string('H+ + OH- = H2O; 1e-14*molar')
    eq5 = Equilibrium.from_string('CO2(aq) = CO2(g);chempy.henry.HenryWithUnits(3.3e-4 * molar / Pa, 2400 * K)')
    assert eq5.reac == {'CO2(aq)': 1}