import pytest
from .._solution import Solution, QuantityDict
from ..util.testing import requires
from ..units import magnitude, units_library, to_unitless, default_units as u
@requires(units_library)
def test_QuantityDict():
    c = QuantityDict(u.molar, {})
    c['H2O'] = 55.4 * u.molar
    with pytest.raises(ValueError):
        c['HCl'] = 3 * u.kg
    with pytest.raises(ValueError):
        QuantityDict(u.molar, {'a': u.mole})
    V = 0.4 * u.dm3
    n = c * V
    assert isinstance(n, QuantityDict)
    assert n.isclose({'H2O': 55.4 * 0.4 * u.mol})
    assert abs(to_unitless(n['H2O'], u.mol) - 55.4 * 0.4) < 1e-14
    c2 = c.rescale(u.mol / u.cm3)
    assert abs(magnitude(c2['H2O']) - 0.0554) < 1e-06