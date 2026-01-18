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
def test_deserialize_dot_z_with_extra_stuff(self):
    primitive = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'versionedobjects', 'versioned_object.version': '1.6.1', 'versioned_object.data': {'foo': 1, 'unexpected_thing': 'foobar'}}
    ser = base.VersionedObjectSerializer()
    obj = ser.deserialize_entity(self.context, primitive)
    self.assertEqual(1, obj.foo)
    self.assertFalse(hasattr(obj, 'unexpected_thing'))
    self.assertEqual('1.6', obj.VERSION)