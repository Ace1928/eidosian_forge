from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_agents_list_routers(self):
    arglist = ['--router', self._testrouter.id]
    verifylist = [('router', self._testrouter.id), ('long', False)]
    attrs = {self._testrouter}
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.routers_hosting_l3_agents.assert_called_once_with(*attrs)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))