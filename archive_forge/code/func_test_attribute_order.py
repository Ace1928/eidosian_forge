import re
import unittest
from wsme import exc
from wsme import types
def test_attribute_order(self):

    class ForcedOrder(object):
        _wsme_attr_order = ('a2', 'a1', 'a3')
        a1 = int
        a2 = int
        a3 = int
    types.register_type(ForcedOrder)
    print(ForcedOrder._wsme_attributes)
    assert ForcedOrder._wsme_attributes[0].key == 'a2'
    assert ForcedOrder._wsme_attributes[1].key == 'a1'
    assert ForcedOrder._wsme_attributes[2].key == 'a3'
    c = gen_class()
    print(c)
    types.register_type(c)
    del c._wsme_attributes
    c.a2 = int
    c.a1 = int
    c.a3 = int
    types.register_type(c)
    assert c._wsme_attributes[0].key == 'a1', c._wsme_attributes[0].key
    assert c._wsme_attributes[1].key == 'a2'
    assert c._wsme_attributes[2].key == 'a3'