from unittest import mock
from unittest.mock import call
import ddt
from osc_lib import exceptions
from openstackclient.network.v2 import network_rbac
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_rbac_create_with_target_all_projects(self):
    arglist = ['--type', self.rbac_policy.object_type, '--action', self.rbac_policy.action, '--target-all-projects', self.rbac_policy.object_id]
    verifylist = [('type', self.rbac_policy.object_type), ('action', self.rbac_policy.action), ('target_all_projects', True), ('rbac_object', self.rbac_policy.object_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_rbac_policy.assert_called_with(**{'object_id': self.rbac_policy.object_id, 'object_type': self.rbac_policy.object_type, 'action': self.rbac_policy.action, 'target_tenant': '*'})