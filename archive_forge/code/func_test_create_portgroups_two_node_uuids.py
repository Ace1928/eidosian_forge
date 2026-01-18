import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
@mock.patch.object(create_resources, 'create_ports', autospec=True)
def test_create_portgroups_two_node_uuids(self, mock_create_ports):
    portgroup = {'name': 'fake', 'node_uuid': 'fake-node-uuid-1', 'ports': ['list of ports']}
    self.client.portgroup.create.side_effect = exc.ClientException('bar')
    errs = create_resources.create_portgroups(self.client, [portgroup], node_uuid='fake-node-uuid-2')
    self.assertFalse(self.client.portgroup.create.called)
    self.assertFalse(mock_create_ports.called)
    self.assertEqual(1, len(errs))
    self.assertIsInstance(errs[0], exc.ClientException)