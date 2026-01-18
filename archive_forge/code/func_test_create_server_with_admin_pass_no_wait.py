import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_server_with_admin_pass_no_wait(self):
    """
        Test that a server with an admin_pass passed returns the password
        """
    admin_pass = self.getUniqueString('password')
    fake_server = fakes.make_fake_server('1234', '', 'BUILD')
    fake_create_server = fakes.make_fake_server('1234', '', 'BUILD', admin_pass=admin_pass)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []}), self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': fake_create_server}, validate=dict(json={'server': {'adminPass': admin_pass, 'flavorRef': 'flavor-id', 'imageRef': 'image-id', 'max_count': 1, 'min_count': 1, 'name': 'server-name', 'networks': 'auto'}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), json={'server': fake_server})])
    self.assertEqual(admin_pass, self.cloud.create_server(name='server-name', image=dict(id='image-id'), flavor=dict(id='flavor-id'), admin_pass=admin_pass)['admin_password'])
    self.assert_calls()