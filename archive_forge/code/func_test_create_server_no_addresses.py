import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch.object(connection.Connection, 'add_ips_to_server')
def test_create_server_no_addresses(self, mock_add_ips_to_server):
    """
        Test that create_server with a wait throws an exception if the
        server doesn't have addresses.
        """
    build_server = fakes.make_fake_server('1234', '', 'BUILD')
    fake_server = fakes.make_fake_server('1234', '', 'ACTIVE', addresses={})
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []}), self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': build_server}, validate=dict(json={'server': {'flavorRef': 'flavor-id', 'imageRef': 'image-id', 'max_count': 1, 'min_count': 1, 'name': 'server-name', 'networks': 'auto'}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': [build_server]}), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': [fake_server]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports'], qs_elements=['device_id=1234']), json={'ports': []}), dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['servers', '1234'])), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), status_code=404)])
    mock_add_ips_to_server.return_value = fake_server
    self.assertRaises(exceptions.SDKException, self.cloud.create_server, 'server-name', {'id': 'image-id'}, {'id': 'flavor-id'}, wait=True)
    self.assert_calls()