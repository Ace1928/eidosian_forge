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
def test_hydration_version_different(self):
    primitive = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'versionedobjects', 'versioned_object.version': '1.2', 'versioned_object.data': {'foo': 1}}
    obj = MyObj.obj_from_primitive(primitive)
    self.assertEqual(obj.foo, 1)
    self.assertEqual('1.2', obj.VERSION)