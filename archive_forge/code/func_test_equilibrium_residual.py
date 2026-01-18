from ..util.testing import requires
from .._equilibrium import equilibrium_residual, solve_equilibrium, _get_rc_interval
from .._util import prodpow
@requires('numpy')
def test_equilibrium_residual():
    c0 = np.array((13.0, 11, 17))
    stoich = np.array((-2, 3, -4))
    K = 3.14
    assert abs(equilibrium_residual(0.1, c0, stoich, K) - (K - (13 - 0.2) ** (-2) * (11 + 0.3) ** 3 * (17 - 0.4) ** (-4))) < 1e-14