from unittest import mock
from unittest.mock import call
import ddt
from osc_lib import exceptions
from openstackclient.network.v2 import network_rbac
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_rbac_create_invalid_type(self):
    arglist = ['--action', self.rbac_policy.action, '--type', 'invalid_type', '--target-project', self.rbac_policy.target_project_id, self.rbac_policy.object_id]
    verifylist = [('action', self.rbac_policy.action), ('type', 'invalid_type'), ('target-project', self.rbac_policy.target_project_id), ('rbac_policy', self.rbac_policy.id)]
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)