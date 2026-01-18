import math
from operator import add
from functools import reduce
import pytest
from chempy import Substance
from chempy.units import (
from ..testing import requires
from ..pyutil import defaultkeydict
from .._expr import (
from ..parsing import parsing_library
@requires(units_library)
def test_Expr_units():
    cv = _get_cv(u.kelvin, u.gram, u.mol)
    R = default_constants.molar_gas_constant.rescale(u.joule / u.mol / u.kelvin)

    def _check(T=273.15 * u.kelvin):
        result = cv['Be']({'temperature': T, 'molar_gas_constant': R}, backend=Backend())
        ref = 0.7342617587256584 * u.joule / u.gram / u.kelvin
        assert abs(to_unitless((result - ref) / ref)) < 1e-10
    _check()
    _check(491.67 * u.rankine)