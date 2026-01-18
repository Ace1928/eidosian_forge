from math import log as ln
from ..electrolytes import A as A_dh, B as B_dh
from ..electrolytes import limiting_log_gamma, _ActivityProductBase, ionic_strength
from ..units import (
from ..util.testing import requires
@requires(units_library)
def test_A__units():
    A20q = A_dh(80.1, 293.15 * u.K, 998.2071 * u.kg / u.m ** 3, b0=1 * u.mol / u.kg, constants=consts)
    assert abs(A20q - 0.50669) < 1e-05