from unittest import mock
from openstack.cloud import inventory
import openstack.config
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch('openstack.config.loader.OpenStackConfig')
@mock.patch('openstack.connection.Connection')
def test_list_hosts(self, mock_cloud, mock_config):
    mock_config.return_value.get_all.return_value = [{}]
    inv = inventory.OpenStackInventory()
    server = dict(id='server_id', name='server_name')
    self.assertIsInstance(inv.clouds, list)
    self.assertEqual(1, len(inv.clouds))
    inv.clouds[0].list_servers.return_value = [server]
    inv.clouds[0].get_openstack_vars.return_value = server
    ret = inv.list_hosts()
    inv.clouds[0].list_servers.assert_called_once_with(detailed=True, all_projects=False)
    self.assertFalse(inv.clouds[0].get_openstack_vars.called)
    self.assertEqual([server], ret)