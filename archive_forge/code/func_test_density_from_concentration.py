from chempy.util.testing import requires
from chempy.units import units_library
@requires('numpy')
def test_density_from_concentration():
    rho = density_from_concentration(1000)
    assert abs(1058.5 - rho) < 0.1