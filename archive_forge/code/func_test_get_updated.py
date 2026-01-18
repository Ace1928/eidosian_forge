from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
def test_get_updated(self):
    self.assertEqual(self.UPDATED_TIMESTAMP, self.extn.get_updated())