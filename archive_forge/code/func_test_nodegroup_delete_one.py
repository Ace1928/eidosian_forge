import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_nodegroup_delete_one(self):
    arglist = ['foo', 'fake-nodegroup']
    verifylist = [('cluster', 'foo'), ('nodegroup', ['fake-nodegroup'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.ng_mock.delete.assert_called_with('foo', 'fake-nodegroup')