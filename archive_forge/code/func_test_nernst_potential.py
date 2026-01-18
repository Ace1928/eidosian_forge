from ..nernst import nernst_potential
from chempy.util.testing import requires
from chempy.units import default_units, default_constants, units_library, allclose
def test_nernst_potential():
    """
    Test cases obtained from textbook examples of Nernst potential in cellular
    membranes. 310K = 37C, typical mammalian cell environment temperature.
    """
    assert abs(1000 * nernst_potential(145, 15, 1, 310) - 60.605) < 0.0001
    assert abs(1000 * nernst_potential(4, 150, 1, 310) - -96.8196) < 0.0001
    assert abs(1000 * nernst_potential(2, 7e-05, 2, 310) - 137.0436) < 0.0001
    assert abs(1000 * nernst_potential(110, 10, -1, 310) - -64.0567) < 0.0001