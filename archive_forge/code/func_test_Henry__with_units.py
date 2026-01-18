from ..henry import Henry, HenryWithUnits
from ..units import units_library, allclose, default_units as u
from ..util.testing import requires
@requires(units_library, 'numpy')
def test_Henry__with_units():
    import numpy as np
    kH_H2 = HenryWithUnits(0.00078 * u.molar / u.atm, 640 * u.K, ref='dean_1992')
    assert allclose(kH_H2.get_kH_at_T(300 * u.K), 0.0007697430323 * u.molar / u.atm)
    kH = kH_H2.get_c_at_T_and_P(np.linspace(297.5, 298.65, 3) * u.K, 0.1 * u.bar)
    assert allclose(kH, 7.7e-05 * u.molar, rtol=1e-05, atol=1e-06 * u.molar)
    kH = kH_H2.get_P_at_T_and_c(298.15 * u.K, np.linspace(0.002, 0.0021, 3) * u.molar)
    assert allclose(kH, 2.65 * u.atm, rtol=1e-05, atol=0.2 * u.bar)