from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
def test_configuration_create(self):
    args = ['cgroup1', '{"param1": 1, "param2": 2}']
    parsed_args = self.check_parser(self.cmd, args, [])
    self.cmd.take_action(parsed_args)
    self.configuration_client.create.assert_called_with('cgroup1', '{"param1": 1, "param2": 2}', description=None, datastore=None, datastore_version=None, datastore_version_number=None)