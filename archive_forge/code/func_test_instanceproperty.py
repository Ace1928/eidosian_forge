from toolz import *
import toolz
import toolz.curried
import pickle
from toolz.utils import raises
def test_instanceproperty():
    p = toolz.functoolz.InstanceProperty(bool)
    assert p.__get__(None) is None
    assert p.__get__(0) is False
    assert p.__get__(1) is True
    p2 = pickle.loads(pickle.dumps(p))
    assert p2.__get__(None) is None
    assert p2.__get__(0) is False
    assert p2.__get__(1) is True