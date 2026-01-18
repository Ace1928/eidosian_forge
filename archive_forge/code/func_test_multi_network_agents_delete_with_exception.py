from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_network_agents_delete_with_exception(self):
    arglist = [self.network_agents[0].id, 'unexist_network_agent']
    verifylist = [('network_agent', [self.network_agents[0].id, 'unexist_network_agent'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    delete_mock_result = [True, exceptions.CommandError]
    self.network_client.delete_agent = mock.Mock(side_effect=delete_mock_result)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 network agents failed to delete.', str(e))
    self.network_client.delete_agent.assert_any_call(self.network_agents[0].id, ignore_missing=False)
    self.network_client.delete_agent.assert_any_call('unexist_network_agent', ignore_missing=False)