import pytest
from .._solution import Solution, QuantityDict
from ..util.testing import requires
from ..units import magnitude, units_library, to_unitless, default_units as u
@requires(units_library)
def test_Solution__dissolve():
    s1, s2 = _get_s1_s2()
    s4 = (s1 + s2).dissolve({'CH3OH': 1 * u.gram})
    mw = 12.011 + 4 * 1.008 + 15.999
    assert abs(s4.concentrations['CH3OH'] - (0.325 + 1 / mw / 0.4) * u.molar) < 1e-07