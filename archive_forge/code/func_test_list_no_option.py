from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_list_no_option(self):
    arglist = []
    verifylist = [('long', False), ('available', False), ('unavailable', False), ('used', False), ('unused', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.network_segment_ranges.assert_called_once_with()
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))