import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
@mock.patch.object(create_resources, 'create_ports', autospec=True)
def test_create_portgroups_exception(self, mock_create_ports):
    portgroup = {'name': 'fake', 'ports': ['list of ports']}
    portgroup_posted = {'name': 'fake', 'node_uuid': 'fake-node-uuid'}
    self.client.portgroup.create.side_effect = exc.ClientException('bar')
    errs = create_resources.create_portgroups(self.client, [portgroup], node_uuid='fake-node-uuid')
    self.client.portgroup.create.assert_called_once_with(**portgroup_posted)
    self.assertFalse(mock_create_ports.called)
    self.assertEqual(1, len(errs))
    self.assertIsInstance(errs[0], exc.ClientException)