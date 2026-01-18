import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_server_fails(self):
    """
        Test that delete_server raises non-404 exceptions
        """
    server = fakes.make_fake_server('1212', 'speedy', 'ACTIVE')
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'speedy']), status_code=404), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail'], qs_elements=['name=speedy']), json={'servers': [server]}), dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['servers', '1212']), status_code=400)])
    self.assertRaises(exceptions.SDKException, self.cloud.delete_server, 'speedy', wait=False)
    self.assert_calls()