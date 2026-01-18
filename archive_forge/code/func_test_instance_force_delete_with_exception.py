from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
@mock.patch.object(utils, 'find_resource')
def test_instance_force_delete_with_exception(self, mock_find):
    args = ['fakeinstance']
    parsed_args = self.check_parser(self.cmd, args, [])
    mock_find.return_value = args[0]
    self.instance_client.delete.side_effect = exceptions.CommandError
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)