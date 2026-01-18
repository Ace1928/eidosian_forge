import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_comparable_objects(self):

    class NonVersionedObject(object):
        pass
    obj1 = MyComparableObj(foo=1)
    obj2 = MyComparableObj(foo=1)
    obj3 = MyComparableObj(foo=2)
    obj4 = NonVersionedObject()
    self.assertTrue(obj1 == obj2)
    self.assertFalse(obj1 == obj3)
    self.assertFalse(obj1 == obj4)
    self.assertNotEqual(obj1, None)