import dill
from dill import objects
from dill import load_types
def test_frame_related():
    g = _g(1)
    f = g.gi_frame
    e, t = _f()
    _is = lambda ok: ok
    ok = dill.pickles(f)
    if verbose:
        print('%s: %s, %s' % (ok, type(f), f))
    assert not ok
    ok = dill.pickles(g)
    if verbose:
        print('%s: %s, %s' % (ok, type(g), g))
    assert _is(not ok)
    ok = dill.pickles(t)
    if verbose:
        print('%s: %s, %s' % (ok, type(t), t))
    assert not ok
    ok = dill.pickles(e)
    if verbose:
        print('%s: %s, %s' % (ok, type(e), e))
    assert ok
    if verbose:
        print('')