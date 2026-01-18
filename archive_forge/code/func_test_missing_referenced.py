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
def test_missing_referenced(self):
    """Ensure a missing child object is highlighted."""

    @base.VersionedObjectRegistry.register
    class TestObjectFoo(base.VersionedObject):
        VERSION = '1.23'
        fields = {'child': fields.ObjectField('TestChildBar')}
    exc = self.assertRaises(exception.UnregisteredSubobject, base.obj_tree_get_versions, 'TestObjectFoo')
    self.assertIn('TestChildBar is referenced by TestObjectFoo', exc.format_message())