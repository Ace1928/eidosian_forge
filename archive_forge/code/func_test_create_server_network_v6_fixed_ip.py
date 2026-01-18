import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_server_network_v6_fixed_ip(self):
    """
        Verify that if 'v6-fixed-ip' is supplied in nics, we pass it to
        networks appropriately.
        """
    network = {'id': 'network-id', 'name': 'network-name'}
    fixed_ip = 'fe80::28da:5fff:fe57:13ed'
    build_server = fakes.make_fake_server('1234', '', 'BUILD')
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': build_server}, validate=dict(json={'server': {'flavorRef': 'flavor-id', 'imageRef': 'image-id', 'max_count': 1, 'min_count': 1, 'networks': [{'fixed_ip': fixed_ip}], 'name': 'server-name'}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), json={'server': build_server}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': [network]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets']), json={'subnets': []})])
    self.cloud.create_server('server-name', dict(id='image-id'), dict(id='flavor-id'), nics=[{'fixed_ip': fixed_ip}])
    self.assert_calls()