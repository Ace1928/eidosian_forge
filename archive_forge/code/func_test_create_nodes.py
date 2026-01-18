import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
@mock.patch.object(create_resources, 'create_portgroups', autospec=True)
@mock.patch.object(create_resources, 'create_ports', autospec=True)
def test_create_nodes(self, mock_create_ports, mock_create_portgroups):
    node = {'driver': 'fake', 'ports': ['list of ports'], 'portgroups': ['list of portgroups']}
    self.client.node.create.return_value = mock.Mock(uuid='uuid')
    self.assertEqual([], create_resources.create_nodes(self.client, [node]))
    self.client.node.create.assert_called_once_with(driver='fake')
    mock_create_ports.assert_called_once_with(self.client, ['list of ports'], node_uuid='uuid')
    mock_create_portgroups.assert_called_once_with(self.client, ['list of portgroups'], node_uuid='uuid')