import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_import_object_ns_raise_import_error_in_init(self):
    self.assertRaises(ImportError, importutils.import_object_ns, 'tests2', 'oslo_utils.tests.fake.FakeDriver3')