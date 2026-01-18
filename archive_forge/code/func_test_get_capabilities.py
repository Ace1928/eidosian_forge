from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.capabilities import Capabilities
def test_get_capabilities(self):
    capabilities = cs.capabilities.get('host')
    cs.assert_called('GET', '/capabilities/host')
    self.assertEqual(FAKE_CAPABILITY, capabilities._info)
    self._assert_request_id(capabilities)