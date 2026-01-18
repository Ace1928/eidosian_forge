import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_nodegroup_show_pass(self):
    arglist = ['fake-cluster', 'fake-nodegroup']
    verifylist = [('cluster', 'fake-cluster'), ('nodegroup', 'fake-nodegroup')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.ng_mock.get.assert_called_with('fake-cluster', 'fake-nodegroup')
    self.assertEqual(osc_nodegroups.NODEGROUP_ATTRIBUTES, columns)
    self.assertEqual(self.data, data)