from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_policy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_no_default(self):
    arglist = [self.new_qos_policy.name, '--no-default']
    verifylist = [('project', None), ('name', self.new_qos_policy.name), ('default', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_qos_policy.assert_called_once_with(**{'name': self.new_qos_policy.name, 'is_default': False})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)