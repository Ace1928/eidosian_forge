import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_nodegroup_list_ok(self):
    arglist = ['fake-cluster']
    verifylist = [('cluster', 'fake-cluster'), ('limit', None), ('sort_key', None), ('sort_dir', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.ng_mock.list.assert_called_with('fake-cluster', limit=None, sort_dir=None, sort_key=None, role=None)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))