import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
def test_fingerprint(self):
    field1 = fields.ListOfEnumField(valid_values=['foo', 'bar'])
    field2 = fields.ListOfEnumField(valid_values=['foo', 'bar1'])
    self.assertNotEqual(str(field1), str(field2))