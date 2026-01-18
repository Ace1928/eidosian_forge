from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_icmp_code_zero(self):
    self._setup_security_group_rule({'port_range_min': 15, 'port_range_max': 0, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
    arglist = ['--protocol', self._security_group_rule.protocol, '--icmp-type', str(self._security_group_rule.port_range_min), '--icmp-code', str(self._security_group_rule.port_range_max), self._security_group.id]
    verifylist = [('protocol', self._security_group_rule.protocol), ('icmp_code', self._security_group_rule.port_range_max), ('icmp_type', self._security_group_rule.port_range_min), ('group', self._security_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.expected_columns, columns)
    self.assertEqual(self.expected_data, data)