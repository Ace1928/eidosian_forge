from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch.object(meta, 'get_server_external_ipv6')
@mock.patch.object(meta, 'get_server_external_ipv4')
def test_ipv4_hostvars(self, mock_get_server_external_ipv4, mock_get_server_external_ipv6):
    mock_get_server_external_ipv4.return_value = PUBLIC_V4
    mock_get_server_external_ipv6.return_value = PUBLIC_V6
    fake_cloud = FakeCloud()
    fake_cloud.force_ipv4 = True
    hostvars = meta.get_hostvars_from_server(fake_cloud, meta.obj_to_munch(standard_fake_server))
    self.assertEqual(PUBLIC_V4, hostvars['interface_ip'])
    self.assertEqual('', hostvars['public_v6'])