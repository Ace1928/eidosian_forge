from unittest import mock
from openstack.cloud import inventory
import openstack.config
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch('openstack.config.loader.OpenStackConfig')
@mock.patch('openstack.connection.Connection')
def test__init_one_cloud(self, mock_cloud, mock_config):
    mock_config.return_value.get_one.return_value = [{}]
    inv = inventory.OpenStackInventory(cloud='supercloud')
    mock_config.assert_called_once_with(config_files=openstack.config.loader.CONFIG_FILES)
    self.assertIsInstance(inv.clouds, list)
    self.assertEqual(1, len(inv.clouds))
    self.assertFalse(mock_config.return_value.get_all.called)
    mock_config.return_value.get_one.assert_called_once_with('supercloud')