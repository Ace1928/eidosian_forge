from unittest import mock
from openstack.cloud import inventory
import openstack.config
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch('openstack.config.loader.OpenStackConfig')
@mock.patch('openstack.connection.Connection')
def test_list_hosts_no_detail(self, mock_cloud, mock_config):
    mock_config.return_value.get_all.return_value = [{}]
    inv = inventory.OpenStackInventory()
    server = self.cloud._normalize_server(fakes.make_fake_server('1234', 'test', 'ACTIVE', addresses={}))
    self.assertIsInstance(inv.clouds, list)
    self.assertEqual(1, len(inv.clouds))
    inv.clouds[0].list_servers.return_value = [server]
    inv.list_hosts(expand=False)
    inv.clouds[0].list_servers.assert_called_once_with(detailed=False, all_projects=False)
    self.assertFalse(inv.clouds[0].get_openstack_vars.called)