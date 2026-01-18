import argparse
from unittest import mock
import uuid
from osc_lib import exceptions
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient.osc import utils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_groups as osc_share_groups
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_list_share_groups_all_projects(self):
    all_tenants_list = self.column_headers.copy()
    all_tenants_list.append('Project ID')
    list_values = (oscutils.get_dict_properties(s._info, all_tenants_list) for s in self.share_groups_list)
    arglist = ['--all-projects']
    verifylist = [('all_projects', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.groups_mock.list.assert_called_with(search_opts={'all_tenants': True, 'name': None, 'status': None, 'share_server_id': None, 'share_group_type': None, 'snapshot': None, 'host': None, 'share_network': None, 'project_id': None, 'limit': None, 'offset': None, 'name~': None, 'description~': None, 'description': None})
    self.assertEqual(all_tenants_list, columns)
    self.assertEqual(list(list_values), list(data))