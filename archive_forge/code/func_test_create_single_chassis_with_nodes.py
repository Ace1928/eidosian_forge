import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
def test_create_single_chassis_with_nodes(self):
    params = {'nodes': ['some nodes here']}
    self.client.chassis.create.return_value = mock.Mock(uuid='uuid')
    self.assertEqual(('uuid', None), create_resources.create_single_chassis(self.client, **params))
    self.client.chassis.create.assert_called_once_with()