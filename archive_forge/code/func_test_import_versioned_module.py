import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_import_versioned_module(self):
    v2 = importutils.import_versioned_module('oslo_utils.tests.fake', 2)
    self.assertEqual(sys.modules['oslo_utils.tests.fake.v2'], v2)
    dummpy = importutils.import_versioned_module('oslo_utils.tests.fake', 2, 'dummpy')
    self.assertEqual(sys.modules['oslo_utils.tests.fake.v2.dummpy'], dummpy)