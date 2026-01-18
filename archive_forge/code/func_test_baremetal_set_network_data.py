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
def test_baremetal_set_network_data(self, mock_handle, mock_stdin):
    self.cmd.log = mock.Mock(autospec=True)
    network_data_string = '{"a": ["b"]}'
    expected_network_data = {'a': ['b']}
    mock_handle.return_value = expected_network_data.copy()
    arglist = ['node_uuid', '--network-data', network_data_string]
    verifylist = [('nodes', ['node_uuid']), ('network_data', network_data_string)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/network_data', 'value': expected_network_data, 'op': 'add'}], reset_interfaces=None)