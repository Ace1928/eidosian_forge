import uuid
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_server_console_no_console(self):
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri='{endpoint}/servers/{id}/action'.format(endpoint=fakes.COMPUTE_ENDPOINT, id=self.server_id), status_code=400, validate=dict(json={'os-getConsoleOutput': {}}))])
    self.assertEqual('', self.cloud.get_server_console(self.server))
    self.assert_calls()