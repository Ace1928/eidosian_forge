from .. import Backend
import pytest
@pytest.mark.parametrize('key', backends)
def test_cse_with_symbols(key):
    be = Backend(key)
    x = be.Symbol('x')
    exprs = [x ** 2, 1 / (1 + x ** 2), be.log(x + 2), be.exp(x + 2)]
    subs_cses, cse_exprs = be.cse(exprs, symbols=be.numbered_symbols('y'))
    subs, cses = zip(*subs_cses)
    assert subs[0] == be.Symbol('y0')
    assert subs[1] == be.Symbol('y1')
    assert _inverse_cse(subs_cses, cse_exprs) == exprs