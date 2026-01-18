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
def test_hydration_additional_unexpected_stuff(self):
    primitive = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'versionedobjects', 'versioned_object.version': '1.5.1', 'versioned_object.data': {'foo': 1, 'unexpected_thing': 'foobar'}}
    obj = MyObj.obj_from_primitive(primitive)
    self.assertEqual(1, obj.foo)
    self.assertFalse(hasattr(obj, 'unexpected_thing'))
    self.assertEqual('1.5.1', obj.VERSION)