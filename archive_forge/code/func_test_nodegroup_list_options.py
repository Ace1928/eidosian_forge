import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_nodegroup_list_options(self):
    arglist = ['fake-cluster', '--limit', '1', '--sort-key', 'key', '--sort-dir', 'asc']
    verifylist = [('cluster', 'fake-cluster'), ('limit', 1), ('sort_key', 'key'), ('sort_dir', 'asc')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.ng_mock.list.assert_called_with('fake-cluster', limit=1, sort_dir='asc', sort_key='key', role=None)