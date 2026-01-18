from toolz import *
import toolz
import toolz.curried
import pickle
from toolz.utils import raises
def test_juxt():
    f = juxt(str, int, bool)
    g = pickle.loads(pickle.dumps(f))
    assert f(1) == g(1)
    assert f.funcs == g.funcs