import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
@mock.patch.object(create_resources, 'create_ports', autospec=True)
def test_create_portgroups(self, mock_create_ports):
    portgroup = {'name': 'fake', 'ports': ['list of ports']}
    portgroup_posted = {'name': 'fake', 'node_uuid': 'fake-node-uuid'}
    self.client.portgroup.create.return_value = mock.Mock(uuid='uuid')
    self.assertEqual([], create_resources.create_portgroups(self.client, [portgroup], node_uuid='fake-node-uuid'))
    self.client.portgroup.create.assert_called_once_with(**portgroup_posted)
    mock_create_ports.assert_called_once_with(self.client, ['list of ports'], node_uuid='fake-node-uuid', portgroup_uuid='uuid')