import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch.object(connection.Connection, 'wait_for_server')
def test_create_server_with_admin_pass_wait(self, mock_wait):
    """
        Test that a server with an admin_pass passed returns the password
        """
    admin_pass = self.getUniqueString('password')
    fake_server = fakes.make_fake_server('1234', '', 'BUILD')
    fake_server_with_pass = fakes.make_fake_server('1234', '', 'BUILD', admin_pass=admin_pass)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []}), self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': fake_server_with_pass}, validate=dict(json={'server': {'flavorRef': 'flavor-id', 'imageRef': 'image-id', 'max_count': 1, 'min_count': 1, 'adminPass': admin_pass, 'name': 'server-name', 'networks': 'auto'}}))])
    mock_wait.return_value = server.Server(**fake_server)
    new_server = self.cloud.create_server(name='server-name', image=dict(id='image-id'), flavor=dict(id='flavor-id'), admin_pass=admin_pass, wait=True)
    self.assertTrue(mock_wait.called)
    self.assertEqual(new_server['admin_password'], fake_server_with_pass['adminPass'])
    self.assert_calls()