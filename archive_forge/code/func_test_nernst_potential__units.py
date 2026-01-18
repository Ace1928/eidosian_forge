from ..nernst import nernst_potential
from chempy.util.testing import requires
from chempy.units import default_units, default_constants, units_library, allclose
@requires(units_library)
def test_nernst_potential__units():
    J = default_units.joule
    K = default_units.kelvin
    coulomb = default_units.coulomb
    v = nernst_potential(145, 15, 1, 310 * K, default_constants)
    assert allclose(1000 * v, 60.605 * J / coulomb, rtol=0.0001)