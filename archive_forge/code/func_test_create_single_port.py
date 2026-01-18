import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
def test_create_single_port(self):
    params = {'address': 'fake-address', 'node_uuid': 'fake-node-uuid'}
    self.client.port.create.return_value = mock.Mock(uuid='fake-port-uuid')
    self.assertEqual(('fake-port-uuid', None), create_resources.create_single_port(self.client, **params))
    self.client.port.create.assert_called_once_with(**params)