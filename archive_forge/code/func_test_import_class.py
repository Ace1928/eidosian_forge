import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_import_class(self):
    dt = importutils.import_class('datetime.datetime')
    self.assertEqual(sys.modules['datetime'].datetime, dt)