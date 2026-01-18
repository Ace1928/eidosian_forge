from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_delete_multiple(self):
    arglist = []
    for _network_segment_range in self._network_segment_ranges:
        arglist.append(_network_segment_range.id)
    verifylist = [('network_segment_range', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for _network_segment_range in self._network_segment_ranges:
        calls.append(call(_network_segment_range))
    self.network_client.delete_network_segment_range.assert_has_calls(calls)
    self.assertIsNone(result)