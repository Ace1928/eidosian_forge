from ..henry import Henry, HenryWithUnits
from ..units import units_library, allclose, default_units as u
from ..util.testing import requires
@requires(units_library)
def test_HenryWithUnits():
    kH_H2 = HenryWithUnits(0.00078 * u.molar / u.atm, 640 * u.K, ref='dean_1992')
    Hcp = kH_H2(300 * u.K)
    assert allclose(Hcp, 0.0007697430323 * u.molar / u.atm)