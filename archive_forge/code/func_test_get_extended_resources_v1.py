from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
def test_get_extended_resources_v1(self):
    self.assertEqual({}, self.extn.get_extended_resources('1.0'))