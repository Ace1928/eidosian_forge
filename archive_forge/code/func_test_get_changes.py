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
def test_get_changes(self):
    obj = MyObj()
    self.assertEqual({}, obj.obj_get_changes())
    obj.foo = 123
    self.assertEqual({'foo': 123}, obj.obj_get_changes())
    obj.bar = 'test'
    self.assertEqual({'foo': 123, 'bar': 'test'}, obj.obj_get_changes())
    obj.obj_reset_changes()
    self.assertEqual({}, obj.obj_get_changes())
    timestamp = datetime.datetime(2001, 1, 1, tzinfo=pytz.utc)
    with mock.patch.object(timeutils, 'utcnow') as mock_utcnow:
        mock_utcnow.return_value = timestamp
        obj.timestamp = timeutils.utcnow()
        self.assertEqual({'timestamp': timestamp}, obj.obj_get_changes())
    obj.obj_reset_changes()
    self.assertEqual({}, obj.obj_get_changes())
    timestamp = datetime.datetime(2001, 1, 1)
    with mock.patch.object(timeutils, 'utcnow') as mock_utcnow:
        mock_utcnow.return_value = timestamp
        obj.timestamp = timeutils.utcnow()
        self.assertRaises(TypeError, obj.obj_get_changes())
    obj.obj_reset_changes()
    self.assertEqual({}, obj.obj_get_changes())