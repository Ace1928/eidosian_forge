from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_pools
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_list_share_pools_detail(self):
    detail_columns = ['Name', 'Host', 'Backend', 'Pool', 'Capabilities']
    detail_values = (oscutils.get_dict_properties(pool._info, detail_columns) for pool in self.share_pools)
    arglist = ['--detail']
    verifylist = [('detail', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.pools_mock.list.assert_called_with(detailed=True, search_opts={'host': None, 'backend': None, 'pool': None, 'share_type': None})
    self.assertEqual(detail_columns, columns)
    self.assertEqual(list(detail_values), list(data))