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
def test_do_subobject_backport_with_manifest_old_parent(self):
    child = self.ChildObj(foo=1)
    parent = self.ParentObj(child=child)
    manifest = {'ChildObj': '1.0'}
    parent_primitive = parent.obj_to_primitive(target_version='1.1', version_manifest=manifest)
    child_primitive = parent_primitive['versioned_object.data']['child']
    self.assertEqual('1.0', child_primitive['versioned_object.version'])