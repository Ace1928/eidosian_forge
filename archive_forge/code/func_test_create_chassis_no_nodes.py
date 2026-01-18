import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
@mock.patch.object(create_resources, 'create_nodes', autospec=True)
def test_create_chassis_no_nodes(self, mock_create_nodes):
    chassis = {'description': 'fake'}
    self.client.chassis.create.return_value = mock.Mock(uuid='uuid')
    self.assertEqual([], create_resources.create_chassis(self.client, [chassis]))
    self.client.chassis.create.assert_called_once_with(description='fake')
    self.assertFalse(mock_create_nodes.called)