from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
def test_extension_does_not_exist(self):
    self.assertFalse(extensions.is_extension_supported(self._plugin, 'gordon'))