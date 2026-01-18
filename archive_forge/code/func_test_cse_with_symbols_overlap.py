from .. import Backend
import pytest
@pytest.mark.parametrize('key', backends)
def test_cse_with_symbols_overlap(key):
    be = Backend(key)
    x0, x1, y = map(be.Symbol, 'x0 x1 y'.split())
    exprs = [x0 ** 2, x0 ** 2 + be.exp(y) ** 2 + 3, x1 * be.exp(y), be.sin(x1 * be.exp(y) + 1)]
    subs_cses, cse_exprs = be.cse(exprs)
    assert _inverse_cse(subs_cses, cse_exprs) == exprs