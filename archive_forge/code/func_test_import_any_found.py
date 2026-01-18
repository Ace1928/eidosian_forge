import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_import_any_found(self):
    dt = importutils.import_any('foo.bar', 'datetime')
    self.assertEqual(sys.modules['datetime'], dt)