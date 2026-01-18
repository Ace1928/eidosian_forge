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
def test_dehydration_invalid_version(self):
    obj = MyObj(foo=1)
    obj.obj_reset_changes()
    self.assertRaises(exception.InvalidTargetVersion, obj.obj_to_primitive, target_version='1.7')