import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
def test_create_ports(self):
    port = {'address': 'fake-address'}
    port_with_node_uuid = port.copy()
    port_with_node_uuid.update(node_uuid='fake-node-uuid')
    self.client.port.create.return_value = mock.Mock(uuid='uuid')
    self.assertEqual([], create_resources.create_ports(self.client, [port], 'fake-node-uuid'))
    self.client.port.create.assert_called_once_with(**port_with_node_uuid)