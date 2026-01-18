import dill
from enum import EnumMeta
import sys
from collections import namedtuple
def test_metaclass():

    class metaclass_with_new(type):

        def __new__(mcls, name, bases, ns, **kwds):
            cls = super().__new__(mcls, name, bases, ns, **kwds)
            assert mcls is not None
            assert cls.method(mcls)
            return cls

        def method(cls, mcls):
            return isinstance(cls, mcls)
    l = locals()
    exec('class subclass_with_new(metaclass=metaclass_with_new):\n        def __new__(cls):\n            self = super().__new__(cls)\n            return self', None, l)
    subclass_with_new = l['subclass_with_new']
    assert dill.copy(subclass_with_new())