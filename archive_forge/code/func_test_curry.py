from toolz import *
import toolz
import toolz.curried
import pickle
from toolz.utils import raises
def test_curry():
    f = curry(map)(str)
    g = pickle.loads(pickle.dumps(f))
    assert list(f((1, 2, 3))) == list(g((1, 2, 3)))