from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_network_agents_delete(self):
    arglist = []
    for n in self.network_agents:
        arglist.append(n.id)
    verifylist = [('network_agent', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for n in self.network_agents:
        calls.append(call(n.id, ignore_missing=False))
    self.network_client.delete_agent.assert_has_calls(calls)
    self.assertIsNone(result)