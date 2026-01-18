from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
@mock.patch.object(utils, 'find_resource')
def test_instance_resize_volume(self, mock_find):
    args = ['instance1', '5']
    mock_find.side_effect = ['instance1']
    parsed_args = self.check_parser(self.cmd, args, [])
    result = self.cmd.take_action(parsed_args)
    self.instance_client.resize_volume.assert_called_with('instance1', 5)
    self.assertIsNone(result)