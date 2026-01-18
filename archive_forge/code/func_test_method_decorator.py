import dill
from enum import EnumMeta
import sys
from collections import namedtuple
def test_method_decorator():

    class A(object):

        @classmethod
        def test(cls):
            pass
    a = A()
    res = dill.dumps(a)
    new_obj = dill.loads(res)
    new_obj.__class__.test()