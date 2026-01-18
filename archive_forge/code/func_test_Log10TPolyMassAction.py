import math
import pytest
from chempy import Reaction
from chempy.units import (
from chempy.util._expr import Log10, Constant
from chempy.util.testing import requires
from ..rates import MassAction, Radiolytic
from .._rates import (
def test_Log10TPolyMassAction():
    p = MassAction(10 ** ShiftedTPoly([273.15, 0.7, 0.02, 0.003, 0.0004]))
    r = Reaction({'A': 2, 'B': 1}, {'C': 1}, p, {'B': 1})
    res = p({'A': 11, 'B': 13, 'temperature': 298.15}, reaction=r)
    ref = 10 ** (0.7 + 0.02 * 25 + 0.003 * 25 ** 2 + 0.0004 * 25 ** 3)
    assert abs(res - ref * 13 * 11 ** 2) < 1e-15