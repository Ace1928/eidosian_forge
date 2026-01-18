import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_import_object_ns_full_required_arg_present(self):
    obj = importutils.import_object_ns('tests2', 'oslo_utils.tests.fake.FakeDriver2', first_arg=False)
    self.assertEqual(obj.__class__.__name__, 'FakeDriver2')