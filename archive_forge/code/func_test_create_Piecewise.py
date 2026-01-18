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
def test_create_Piecewise():
    PW = create_Piecewise('T')
    Ha, Sa, Hb, Sb, Ta, Tb = (40000.0, -60, 37000.0, -42, 293.15, 303.15)
    a = MyK([Ha, Sa])
    b = MyK([Hb, Sb])
    pw = PW([273.15, a, 298.15, b, 323.15])
    res_a = pw({'T': Ta})
    res_b = pw({'T': Tb})
    ref_a = math.exp(-(Ha - Ta * Sa) / (MyK.R * Ta))
    ref_b = math.exp(-(Hb - Tb * Sb) / (MyK.R * Tb))
    assert abs(res_a - ref_a) < 1e-14
    assert abs(res_b - ref_b) < 1e-14