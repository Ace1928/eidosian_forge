import math
import pytest
from chempy import Reaction
from chempy.units import (
from chempy.util._expr import Log10, Constant
from chempy.util.testing import requires
from ..rates import MassAction, Radiolytic
from .._rates import (
@requires(units_library)
def test_TPolyRadiolytic__units():

    def _check(r):
        res = r({'doserate': 0.15 * u.gray / u.second, 'density': 0.998 * u.kg / u.decimetre ** 3, 'temperature': 298.15 * u.K})
        ref = 0.15 * 0.998 * 2.1e-07 * u.molar / u.second
        assert abs(to_unitless((res - ref) / ref)) < 1e-15
    _check(Radiolytic(ShiftedTPoly([273.15 * u.K, 1.85e-07 * u.mol / u.joule, 1e-09 * u.mol / u.joule / u.K])))