import math
import pytest
from chempy import Reaction
from chempy.units import (
from chempy.util._expr import Log10, Constant
from chempy.util.testing import requires
from ..rates import MassAction, Radiolytic
from .._rates import (
@requires(units_library)
def test_TPolyInLog10MassAction__units():
    Mps = u.molar / u.second
    kunit = 1 / u.molar ** 2 / u.second
    p = MassAction(Constant(kunit) * ShiftedLog10TPoly([2, 0.3, 0.2, 0.03, 0.004]))
    lgT = Log10('temperature' / Constant(u.K))
    r = Reaction({'A': 2, 'B': 1}, {'C': 1}, p, {'B': 1})
    res = p({'A': 11 * u.molar, 'B': 13 * u.molar, 'temperature': 298.15 * u.K, 'log10_temperature': lgT}, backend=Backend(), reaction=r)
    _T = math.log10(298.15) - 2
    ref = (0.3 + 0.2 * _T + 0.03 * _T ** 2 + 0.004 * _T ** 3) * 13 * 11 ** 2 * Mps
    assert abs(res - ref) < 1e-15 * Mps