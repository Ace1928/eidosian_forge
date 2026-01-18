import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_nodegroup_create_required_args_pass(self):
    """Verifies required arguments."""
    arglist = [self.nodegroup.cluster_id, self.nodegroup.name]
    verifylist = [('cluster', self.nodegroup.cluster_id), ('name', self.nodegroup.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.ng_mock.create.assert_called_with(self.nodegroup.cluster_id, **self._default_args)