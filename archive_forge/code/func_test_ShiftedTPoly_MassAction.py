import math
import pytest
from chempy import Reaction
from chempy.units import (
from chempy.util._expr import Log10, Constant
from chempy.util.testing import requires
from ..rates import MassAction, Radiolytic
from .._rates import (
def test_ShiftedTPoly_MassAction():
    rate = MassAction(ShiftedTPoly([100, 2, 5, 7]))
    assert rate.args[0].args == [100, 2, 5, 7]
    r = Reaction({'A': 2, 'B': 1}, {'P': 1}, rate)
    res = r.rate_expr()({'A': 11, 'B': 13, 'temperature': 273.15}, reaction=r)
    x = 273.15 - 100
    ref = 11 * 11 * 13 * (2 + 5 * x + 7 * x ** 2)
    assert abs((res - ref) / ref) < 1e-14