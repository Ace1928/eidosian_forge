from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.compute.v2 import agent
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_delete_multiple_agents(self):
    arglist = []
    for n in self.fake_agents:
        arglist.append(n.agent_id)
    verifylist = [('id', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for n in self.fake_agents:
        calls.append(call(n.agent_id))
    self.agents_mock.delete.assert_has_calls(calls)
    self.assertIsNone(result)