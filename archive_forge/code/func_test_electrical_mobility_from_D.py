from ..einstein_smoluchowski import electrical_mobility_from_D
from ..units import default_units, allclose, default_constants
from ..util.testing import requires
from ..units import units_library
def test_electrical_mobility_from_D():
    D = 3
    z = -2
    T = 100
    mu = electrical_mobility_from_D(D, z, T)
    e = 1.60217657e-19
    kB = 1.3806488e-23
    ref = z * e * D / (kB * T)
    assert allclose(mu, ref, rtol=1e-05)
    mu2 = electrical_mobility_from_D(3, -2, 100)
    assert allclose(mu2, -2 * 1.60217657e-19 * 3 / 1.3806488e-23 / 100, rtol=1e-05)