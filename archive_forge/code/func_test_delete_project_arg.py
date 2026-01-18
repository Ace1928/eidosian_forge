from unittest import mock
from openstackclient.network.v2 import network_auto_allocated_topology
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
def test_delete_project_arg(self):
    arglist = ['--project', self.project.id]
    verifylist = [('project', self.project.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.delete_auto_allocated_topology.assert_called_once_with(self.project.id)
    self.assertIsNone(result)