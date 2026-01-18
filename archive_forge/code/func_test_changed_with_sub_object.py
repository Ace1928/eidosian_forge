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
def test_changed_with_sub_object(self):

    @base.VersionedObjectRegistry.register
    class ParentObject(base.VersionedObject):
        fields = {'foo': fields.IntegerField(), 'bar': fields.ObjectField('MyObj')}
    obj = ParentObject()
    self.assertEqual(set(), obj.obj_what_changed())
    obj.foo = 1
    self.assertEqual(set(['foo']), obj.obj_what_changed())
    bar = MyObj()
    obj.bar = bar
    self.assertEqual(set(['foo', 'bar']), obj.obj_what_changed())
    obj.obj_reset_changes()
    self.assertEqual(set(), obj.obj_what_changed())
    bar.foo = 1
    self.assertEqual(set(['bar']), obj.obj_what_changed())