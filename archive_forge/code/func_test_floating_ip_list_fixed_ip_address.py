from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_floating_ip_list_fixed_ip_address(self):
    arglist = ['--fixed-ip-address', self.floating_ips[0].fixed_ip_address]
    verifylist = [('fixed_ip_address', self.floating_ips[0].fixed_ip_address)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.ips.assert_called_once_with(**{'fixed_ip_address': self.floating_ips[0].fixed_ip_address})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))