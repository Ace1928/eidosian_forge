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
def test_do_subobject_backport_null_child(self):
    parent = self.ParentObj(child=None)
    parent_primitive = parent.obj_to_primitive()['versioned_object.data']
    version = '1.0'
    compat_func = 'obj_make_compatible_from_manifest'
    with mock.patch.object(self.ChildObj, compat_func) as mock_compat:
        base._do_subobject_backport(version, parent, 'child', parent_primitive)
        self.assertFalse(mock_compat.called, 'obj_make_compatible_from_manifest() should not have been called because the subobject is None.')