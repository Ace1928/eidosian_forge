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
def test_changed_4(self):
    obj = MyObj.query(self.context)
    obj.bar = 'something'
    self.assertEqual(obj.obj_what_changed(), set(['bar']))
    obj.modify_save_modify()
    self.assertEqual(obj.obj_what_changed(), set(['foo', 'rel_object']))
    self.assertEqual(obj.foo, 42)
    self.assertEqual(obj.bar, 'meow')
    self.assertIsInstance(obj.rel_object, MyOwnedObject)