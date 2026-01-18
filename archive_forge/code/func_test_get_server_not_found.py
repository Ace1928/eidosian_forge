import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_server_not_found(self):
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': []})])
    r = self.cloud.get_server('doesNotExist')
    self.assertIsNone(r)
    self.assert_calls()