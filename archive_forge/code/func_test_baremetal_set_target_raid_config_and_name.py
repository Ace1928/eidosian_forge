import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
@mock.patch.object(commonutils, 'get_from_stdin', autospec=True)
@mock.patch.object(commonutils, 'handle_json_or_file_arg', autospec=True)
def test_baremetal_set_target_raid_config_and_name(self, mock_handle, mock_stdin):
    self.cmd.log = mock.Mock(autospec=True)
    target_raid_config_string = '{"raid": "config"}'
    expected_target_raid_config = {'raid': 'config'}
    mock_handle.return_value = expected_target_raid_config.copy()
    arglist = ['node_uuid', '--name', 'xxxxx', '--target-raid-config', target_raid_config_string]
    verifylist = [('nodes', ['node_uuid']), ('name', 'xxxxx'), ('target_raid_config', target_raid_config_string)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.cmd.log.warning.assert_not_called()
    self.assertFalse(mock_stdin.called)
    mock_handle.assert_called_once_with(target_raid_config_string)
    self.baremetal_mock.node.set_target_raid_config.assert_called_once_with('node_uuid', expected_target_raid_config)
    self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/name', 'value': 'xxxxx', 'op': 'add'}], reset_interfaces=None)