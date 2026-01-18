from math import log as ln
from ..electrolytes import A as A_dh, B as B_dh
from ..electrolytes import limiting_log_gamma, _ActivityProductBase, ionic_strength
from ..units import (
from ..util.testing import requires
def test_ActivityProductBase():
    ap = _ActivityProductBase((1, -2, 3), 17, 42)
    assert ap.stoich == (1, -2, 3)
    assert ap.args == (17, 42)