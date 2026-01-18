from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_default_rule(self):
    self._setup_security_group_rule({'protocol': 'tcp', 'port_range_max': 443, 'port_range_min': 443})
    arglist = ['--protocol', 'tcp', '--dst-port', str(self._security_group_rule.port_range_min), self._security_group.id]
    verifylist = [('dst_port', (self._security_group_rule.port_range_min, self._security_group_rule.port_range_max)), ('group', self._security_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'port_range_max': self._security_group_rule.port_range_max, 'port_range_min': self._security_group_rule.port_range_min, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
    self.assertEqual(self.expected_columns, columns)
    self.assertEqual(self.expected_data, data)