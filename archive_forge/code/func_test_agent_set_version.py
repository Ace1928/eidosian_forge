from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.compute.v2 import agent
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_agent_set_version(self):
    arglist = ['1', '--agent-version', 'new-version']
    verifylist = [('id', '1'), ('version', 'new-version')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.agents_mock.update.assert_called_with(parsed_args.id, parsed_args.version, self.fake_agent.url, self.fake_agent.md5hash)
    self.assertIsNone(result)