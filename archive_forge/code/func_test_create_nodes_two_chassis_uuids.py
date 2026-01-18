import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
@mock.patch.object(create_resources, 'create_ports', autospec=True)
def test_create_nodes_two_chassis_uuids(self, mock_create_ports):
    node = {'driver': 'fake', 'ports': ['list of ports'], 'chassis_uuid': 'chassis-uuid-1'}
    errs = create_resources.create_nodes(self.client, [node], chassis_uuid='chassis-uuid-2')
    self.assertFalse(self.client.node.create.called)
    self.assertFalse(mock_create_ports.called)
    self.assertEqual(1, len(errs))
    self.assertIsInstance(errs[0], exc.ClientException)