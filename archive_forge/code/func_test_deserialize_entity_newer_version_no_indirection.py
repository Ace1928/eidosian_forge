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
def test_deserialize_entity_newer_version_no_indirection(self):
    ser = base.VersionedObjectSerializer()
    obj = MyObj()
    obj.VERSION = '1.25'
    primitive = obj.obj_to_primitive()
    self.assertRaises(exception.IncompatibleObjectVersion, ser.deserialize_entity, self.context, primitive)