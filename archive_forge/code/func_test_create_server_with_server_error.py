import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_server_with_server_error(self):
    """
        Test that a server error before we return or begin waiting for the
        server instance spawn raises an exception in create_server.
        """
    build_server = fakes.make_fake_server('1234', '', 'BUILD')
    error_server = fakes.make_fake_server('1234', '', 'ERROR')
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []}), self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': build_server}, validate=dict(json={'server': {'flavorRef': 'flavor-id', 'imageRef': 'image-id', 'max_count': 1, 'min_count': 1, 'name': 'server-name', 'networks': 'auto'}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), json={'server': error_server})])
    self.assertRaises(exceptions.SDKException, self.cloud.create_server, 'server-name', {'id': 'image-id'}, {'id': 'flavor-id'})
    self.assert_calls()