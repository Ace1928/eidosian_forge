import math
import pytest
from chempy import Reaction
from chempy.units import (
from chempy.util._expr import Log10, Constant
from chempy.util.testing import requires
from ..rates import MassAction, Radiolytic
from .._rates import (
def test_TPolyInLog10MassAction():
    p = MassAction(ShiftedLog10TPoly([2, 0.3, 0.2, 0.03, 0.004]))
    r = Reaction({'A': 2, 'B': 1}, {'C': 1}, p, {'B': 1})
    lgT = Log10('temperature')
    lgTref = Log10('Tref')
    res = p({'A': 11, 'B': 13, 'temperature': 298.15, 'log10_temperature': lgT, 'log10_Tref': lgTref}, reaction=r)
    _T = math.log10(298.15) - 2
    ref = 0.3 + 0.2 * _T + 0.03 * _T ** 2 + 0.004 * _T ** 3
    assert abs(res - ref * 13 * 11 ** 2) < 1e-15