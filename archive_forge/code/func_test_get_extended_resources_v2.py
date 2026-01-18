from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
def test_get_extended_resources_v2(self):
    self.assertEqual(dict(list(self.RESOURCE_ATTRIBUTE_MAP.items()) + list(self.SUB_RESOURCE_ATTRIBUTE_MAP.items())), self.extn.get_extended_resources('2.0'))