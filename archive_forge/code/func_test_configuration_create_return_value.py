from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
def test_configuration_create_return_value(self):
    args = ['c-123', '{"max_connections": 5}', '--description', 'test_config', '--datastore', 'mysql', '--datastore-version', '5.6']
    parsed_args = self.check_parser(self.cmd, args, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.values, data)