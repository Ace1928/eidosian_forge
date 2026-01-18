import functools
import dill
def test_functools():
    fp = functools.partial(f, 1, 2)
    gp = functools.partial(g, 1, c=2)
    hp = functools.partial(h, 1, c=2)
    bp = functools.partial(int, base=2)
    assert dill.pickles(fp, safe=True)
    assert dill.pickles(gp, safe=True)
    assert dill.pickles(hp, safe=True)
    assert dill.pickles(bp, safe=True)