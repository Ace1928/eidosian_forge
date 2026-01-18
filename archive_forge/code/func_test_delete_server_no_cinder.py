import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_server_no_cinder(self):
    """
        Test that deleting server works when cinder is not available
        """
    orig_has_service = self.cloud.has_service

    def fake_has_service(service_type):
        if service_type == 'volume':
            return False
        return orig_has_service(service_type)
    self.cloud.has_service = fake_has_service
    server = fakes.make_fake_server('1234', 'porky', 'ACTIVE')
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'porky']), status_code=404), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail'], qs_elements=['name=porky']), json={'servers': [server]}), dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']))])
    self.assertTrue(self.cloud.delete_server('porky', wait=False))
    self.assert_calls()