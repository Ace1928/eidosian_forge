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
def test_get_schema(self):
    schema = self.field.get_schema()
    self.assertEqual(['string'], schema['type'])
    self.assertEqual(False, schema['readonly'])
    pattern = schema['pattern']
    for _, valid_val in self.coerce_good_values:
        self.assertRegex(str(valid_val), pattern)
    invalid_vals = [x for x in self.coerce_bad_values]
    for invalid_val in invalid_vals:
        self.assertNotRegex(str(invalid_val), pattern)