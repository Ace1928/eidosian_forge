import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_server_wait_for_deleted(self):
    """
        Test that delete_server waits for the server to be gone
        """
    server = fakes.make_fake_server('9999', 'wily', 'ACTIVE')
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'wily']), status_code=404), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail'], qs_elements=['name=wily']), json={'servers': [server]}), dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['servers', '9999'])), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '9999']), json={'server': server}), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '9999']), status_code=404)])
    self.assertTrue(self.cloud.delete_server('wily', wait=True))
    self.assert_calls()