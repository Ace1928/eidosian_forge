from math import log as ln
from ..electrolytes import A as A_dh, B as B_dh
from ..electrolytes import limiting_log_gamma, _ActivityProductBase, ionic_strength
from ..units import (
from ..util.testing import requires
def test_ionic_strength():
    assert abs(ionic_strength([0.1, 1.3, 2.1, 0.7], [-1, 2, -3, 4], warn=False) - 17.7) < 1e-14