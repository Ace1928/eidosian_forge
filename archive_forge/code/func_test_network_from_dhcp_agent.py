from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_from_dhcp_agent(self):
    arglist = ['--dhcp', self.agent.id, self.net.id]
    verifylist = [('dhcp', True), ('agent_id', self.agent.id), ('network', self.net.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.network_client.remove_dhcp_agent_from_network.assert_called_once_with(self.agent, self.net)