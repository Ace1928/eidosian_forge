import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_server_with_timeout(self):
    """
        Test that a timeout while waiting for the server to spawn raises an
        exception in create_server.
        """
    fake_server = fakes.make_fake_server('1234', '', 'BUILD')
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []}), self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': fake_server}, validate=dict(json={'server': {'flavorRef': 'flavor-id', 'imageRef': 'image-id', 'max_count': 1, 'min_count': 1, 'name': 'server-name', 'networks': 'auto'}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': [fake_server]})])
    self.assertRaises(exceptions.ResourceTimeout, self.cloud.create_server, 'server-name', dict(id='image-id'), dict(id='flavor-id'), wait=True, timeout=0.01)
    self.assert_calls(do_count=False)