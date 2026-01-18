import math
import pytest
from chempy import Reaction
from chempy.units import (
from chempy.util._expr import Log10, Constant
from chempy.util.testing import requires
from ..rates import MassAction, Radiolytic
from .._rates import (
@requires(units_library)
def test_TPiecewise():
    expr0 = ShiftedTPoly([273.15 * u.K, 10 / u.molar / u.s, 0.1 / u.molar / u.s / u.K])
    expr1 = ShiftedTPoly([298.15 * u.K, 12.5 / u.molar / u.s, 0 / u.molar / u.s / u.K, 2 / u.molar / u.s / u.K ** 2])
    pwma = MassAction(TPiecewise([273.15 * u.K, expr0, 298.15 * u.K, expr1, 373.15 * u.K]))
    r = Reaction({'e-(aq)': 2}, {'H2': 1, 'OH-': 2}, inact_reac={'H2O': 2})
    res0 = pwma({'temperature': 293.15 * u.K, 'e-(aq)': 1e-13 * u.molar}, reaction=r)
    ref0 = 12 * 1e-26 * u.molar / u.s
    assert allclose(res0, ref0)
    assert not allclose(res0, 2 * ref0)
    res1 = pwma({'temperature': 300.15 * u.K, 'e-(aq)': 2e-13 * u.molar}, reaction=r)
    ref1 = 20.5 * 4e-26 * u.molar / u.s
    assert allclose(res1, ref1)
    assert not allclose(res1, ref1 / 2)