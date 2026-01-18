from toolz import *
import toolz
import toolz.curried
import pickle
from toolz.utils import raises
def test_curried_bad_qualname():

    @toolz.curry
    class Bad(object):
        __qualname__ = 'toolz.functoolz.not.a.valid.path'
    assert raises(pickle.PicklingError, lambda: pickle.dumps(Bad))