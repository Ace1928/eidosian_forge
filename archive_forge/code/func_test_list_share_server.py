from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_list_share_server(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.servers_mock.list.assert_called_with(search_opts={'status': None, 'host': None, 'project_id': None})
    self.assertEqual(self.columns, columns)
    self.assertEqual(list(self.values), list(data))