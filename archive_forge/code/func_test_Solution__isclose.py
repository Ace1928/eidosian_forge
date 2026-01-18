import pytest
from .._solution import Solution, QuantityDict
from ..util.testing import requires
from ..units import magnitude, units_library, to_unitless, default_units as u
@requires(units_library)
def test_Solution__isclose():
    s1, s2 = _get_s1_s2()
    s3 = s1 + s2
    assert s3.concentrations.isclose({'CH3OH': 0.325 * u.molar, 'Na+': 0.0015 * u.molar, 'Cl-': 0.0015 * u.molar})