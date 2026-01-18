from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_remote_address_group(self):
    self._setup_security_group_rule({'protocol': 'icmp', 'remote_address_group_id': self._address_group.id})
    arglist = ['--protocol', 'icmp', '--remote-address-group', self._address_group.name, self._security_group.id]
    verifylist = [('remote_address_group', self._address_group.name), ('group', self._security_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'protocol': self._security_group_rule.protocol, 'remote_address_group_id': self._address_group.id, 'security_group_id': self._security_group.id})
    self.assertEqual(self.expected_columns, columns)
    self.assertEqual(self.expected_data, data)