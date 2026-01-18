from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_transfers as osc_share_transfers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_list_transfers(self):
    arglist = ['--detailed']
    verifylist = [('detailed', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.transfers_mock.list.assert_called_with(detailed=1, search_opts={'all_tenants': False, 'id': None, 'name': None, 'limit': None, 'offset': None, 'resource_type': None, 'resource_id': None, 'source_project_id': None}, sort_key=None, sort_dir=None)
    self.assertEqual(COLUMNS, columns)
    self.assertEqual(list(self.values), list(data))