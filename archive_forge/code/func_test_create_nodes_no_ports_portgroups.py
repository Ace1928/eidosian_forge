import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
@mock.patch.object(create_resources, 'create_portgroups', autospec=True)
@mock.patch.object(create_resources, 'create_ports', autospec=True)
def test_create_nodes_no_ports_portgroups(self, mock_create_ports, mock_create_portgroups):
    node = {'driver': 'fake'}
    self.client.node.create.return_value = mock.Mock(uuid='uuid')
    self.assertEqual([], create_resources.create_nodes(self.client, [node]))
    self.client.node.create.assert_called_once_with(driver='fake')
    self.assertFalse(mock_create_ports.called)
    self.assertFalse(mock_create_portgroups.called)