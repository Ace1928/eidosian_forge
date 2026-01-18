from unittest import mock
from unittest.mock import call
import ddt
from osc_lib import exceptions
from openstackclient.network.v2 import network_rbac
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_rbac_list_action_opt(self):
    arglist = ['--action', self.rbac_policies[0].action]
    verifylist = [('action', self.rbac_policies[0].action)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.rbac_policies.assert_called_with(**{'action': self.rbac_policies[0].action})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))