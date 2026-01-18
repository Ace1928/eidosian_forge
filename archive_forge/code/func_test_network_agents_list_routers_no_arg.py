from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_agents_list_routers_no_arg(self):
    arglist = ['--routers']
    verifylist = []
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)