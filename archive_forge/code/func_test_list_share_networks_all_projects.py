from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_list_share_networks_all_projects(self):
    all_tenants_list = COLUMNS.copy()
    all_tenants_list.append('Project ID')
    self.expected_search_opts.update({'all_tenants': True})
    list_values = (oscutils.get_dict_properties(s._info, all_tenants_list) for s in self.share_networks_list)
    arglist = ['--all-projects']
    verifylist = [('all_projects', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.share_networks_mock.list.assert_called_once_with(search_opts=self.expected_search_opts)
    self.assertEqual(all_tenants_list, columns)
    self.assertEqual(list(list_values), list(data))