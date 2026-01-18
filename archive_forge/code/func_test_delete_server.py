import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_server(self):
    """
        Test that server delete is called when wait=False
        """
    server = fakes.make_fake_server('1234', 'daffy', 'ACTIVE')
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'daffy']), status_code=404), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail'], qs_elements=['name=daffy']), json={'servers': [server]}), dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']))])
    self.assertTrue(self.cloud.delete_server('daffy', wait=False))
    self.assert_calls()