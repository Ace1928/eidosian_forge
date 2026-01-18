import re
import unittest
from wsme import exc
from wsme import types
def test_cross_referenced_types(self):

    class A(object):
        b = types.wsattr('B')

    class B(object):
        a = A
    types.register_type(A)
    types.register_type(B)
    assert A.b.datatype is B