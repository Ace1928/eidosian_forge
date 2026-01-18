from unittest.mock import call
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group_snapshot
def test_consistency_group_snapshot_list_with_long(self):
    arglist = ['--long']
    verifylist = [('all_projects', False), ('long', True), ('status', None), ('consistency_group', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    search_opts = {'all_tenants': False, 'status': None, 'consistencygroup_id': None}
    self.cgsnapshots_mock.list.assert_called_once_with(detailed=True, search_opts=search_opts)
    self.assertEqual(self.columns_long, columns)
    self.assertEqual(self.data_long, list(data))