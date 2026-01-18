from .. import Backend
import pytest
@pytest.mark.parametrize('key', backends)
def test_moot_cse(key):
    be = Backend(key)
    x, y = map(be.Symbol, 'xy')
    exprs = [x ** 2 + y ** 2, y]
    subs_cses, cse_exprs = be.cse(exprs)
    assert not subs_cses
    assert _inverse_cse(subs_cses, cse_exprs) == exprs