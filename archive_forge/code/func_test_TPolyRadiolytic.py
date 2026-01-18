import math
import pytest
from chempy import Reaction
from chempy.units import (
from chempy.util._expr import Log10, Constant
from chempy.util.testing import requires
from ..rates import MassAction, Radiolytic
from .._rates import (
def test_TPolyRadiolytic():
    tpr = Radiolytic(ShiftedTPoly([273.15, 1.85e-07, 1e-09]))
    res = tpr({'doserate': 0.15, 'density': 0.998, 'temperature': 298.15})
    assert abs(res - 0.15 * 0.998 * 2.1e-07) < 1e-15