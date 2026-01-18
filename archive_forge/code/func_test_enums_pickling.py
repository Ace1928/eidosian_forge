import pickle
from ...types.enum import Enum
def test_enums_pickling():
    a = PickleEnum.A
    pickled = pickle.dumps(a)
    restored = pickle.loads(pickled)
    assert type(a) is type(restored)
    assert a == restored
    assert a.value == restored.value
    assert a.name == restored.name
    b = PickleEnum.B
    pickled = pickle.dumps(b)
    restored = pickle.loads(pickled)
    assert type(a) is type(restored)
    assert b == restored
    assert b.value == restored.value
    assert b.name == restored.name