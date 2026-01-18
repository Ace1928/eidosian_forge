import math
import pytest
from chempy import Reaction
from chempy.units import (
from chempy.util._expr import Log10, Constant
from chempy.util.testing import requires
from ..rates import MassAction, Radiolytic
from .._rates import (
def test_TPiecewisePolyMassAction():
    tp1 = TPoly([10, 0.1])
    tp2 = ShiftedTPoly([273.15, 37.315, -0.1])
    pwp = MassAction(TPiecewise([0, tp1, 273.15, tp2, 373.15]))
    r = Reaction({'A': 2, 'B': 1}, {'C': 1}, inact_reac={'B': 1})
    res1 = pwp({'A': 11, 'B': 13, 'temperature': 198.15}, reaction=r)
    ref1 = 11 * 11 * 13 * 29.815
    assert abs((res1 - ref1) / ref1) < 1e-14
    res2 = pwp({'A': 11, 'B': 13, 'temperature': 298.15}, reaction=r)
    ref2 = 11 * 11 * 13 * (37.315 - 25 * 0.1)
    assert abs((res2 - ref2) / ref2) < 1e-14
    with pytest.raises(ValueError):
        pwp({'A': 11, 'B': 13, 'temperature': 398.15}, reaction=r)