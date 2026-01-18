from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch.object(meta, 'get_server_external_ipv6')
@mock.patch.object(meta, 'get_server_external_ipv4')
def test_basic_hostvars(self, mock_get_server_external_ipv4, mock_get_server_external_ipv6):
    mock_get_server_external_ipv4.return_value = PUBLIC_V4
    mock_get_server_external_ipv6.return_value = PUBLIC_V6
    hostvars = meta.get_hostvars_from_server(FakeCloud(), self.cloud._normalize_server(meta.obj_to_munch(standard_fake_server)))
    self.assertNotIn('links', hostvars)
    self.assertEqual(PRIVATE_V4, hostvars['private_v4'])
    self.assertEqual(PUBLIC_V4, hostvars['public_v4'])
    self.assertEqual(PUBLIC_V6, hostvars['public_v6'])
    self.assertEqual(PUBLIC_V6, hostvars['interface_ip'])
    self.assertEqual('RegionOne', hostvars['region'])
    self.assertEqual('_test_cloud_', hostvars['cloud'])
    self.assertIn('location', hostvars)
    self.assertEqual('_test_cloud_', hostvars['location']['cloud'])
    self.assertEqual('RegionOne', hostvars['location']['region_name'])
    self.assertEqual(fakes.PROJECT_ID, hostvars['location']['project']['id'])
    self.assertEqual('test-image-name', hostvars['image']['name'])
    self.assertEqual(standard_fake_server['image']['id'], hostvars['image']['id'])
    self.assertNotIn('links', hostvars['image'])
    self.assertEqual(standard_fake_server['flavor']['id'], hostvars['flavor']['id'])
    self.assertEqual('test-flavor-name', hostvars['flavor']['name'])
    self.assertNotIn('links', hostvars['flavor'])
    self.assertEqual([], hostvars['volumes'])