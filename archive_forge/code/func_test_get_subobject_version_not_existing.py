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
def test_get_subobject_version_not_existing(self):
    self.assertRaises(exception.TargetBeforeSubobjectExistedException, base._get_subobject_version, '1.0', self.rels, self.backport_mock)