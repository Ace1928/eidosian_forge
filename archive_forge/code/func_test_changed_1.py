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
def test_changed_1(self):
    obj = MyObj.query(self.context)
    obj.foo = 123
    self.assertEqual(obj.obj_what_changed(), set(['foo']))
    obj._update_test()
    self.assertEqual(obj.obj_what_changed(), set(['foo', 'bar']))
    self.assertEqual(obj.foo, 123)