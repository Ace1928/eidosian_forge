from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_agents_list_host(self):
    arglist = ['--host', self.network_agents[0].host]
    verifylist = [('host', self.network_agents[0].host)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.agents.assert_called_once_with(**{'host': self.network_agents[0].host})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))