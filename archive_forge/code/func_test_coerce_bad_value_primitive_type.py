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
def test_coerce_bad_value_primitive_type(self):
    ex = self.assertRaises(ValueError, self.field.coerce, 'obj', 'attr', [{}])
    self.assertEqual('An object of type TestableObject is required in field attr, not a list', str(ex))