import re
import unittest
from wsme import exc
from wsme import types
def test_named_attribute(self):

    class ABCDType(object):
        a_list = types.wsattr([int], name='a.list')
        astr = str
    types.register_type(ABCDType)
    assert len(ABCDType._wsme_attributes) == 2
    attrs = ABCDType._wsme_attributes
    assert attrs[0].key == 'a_list', attrs[0].key
    assert attrs[0].name == 'a.list', attrs[0].name
    assert attrs[1].key == 'astr', attrs[1].key
    assert attrs[1].name == 'astr', attrs[1].name