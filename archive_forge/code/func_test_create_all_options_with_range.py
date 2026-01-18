from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip_port_forwarding
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_all_options_with_range(self):
    arglist = ['--port', self.new_port_forwarding_with_ranges.internal_port_id, '--internal-protocol-port', self.new_port_forwarding_with_ranges.internal_port_range, '--external-protocol-port', self.new_port_forwarding_with_ranges.external_port_range, '--protocol', self.new_port_forwarding_with_ranges.protocol, self.new_port_forwarding_with_ranges.floatingip_id, '--internal-ip-address', self.new_port_forwarding_with_ranges.internal_ip_address, '--description', self.new_port_forwarding_with_ranges.description]
    verifylist = [('port', self.new_port_forwarding_with_ranges.internal_port_id), ('internal_protocol_port', self.new_port_forwarding_with_ranges.internal_port_range), ('external_protocol_port', self.new_port_forwarding_with_ranges.external_port_range), ('protocol', self.new_port_forwarding_with_ranges.protocol), ('floating_ip', self.new_port_forwarding_with_ranges.floatingip_id), ('internal_ip_address', self.new_port_forwarding_with_ranges.internal_ip_address), ('description', self.new_port_forwarding_with_ranges.description)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_floating_ip_port_forwarding.assert_called_once_with(self.new_port_forwarding.floatingip_id, **{'external_port_range': self.new_port_forwarding_with_ranges.external_port_range, 'internal_ip_address': self.new_port_forwarding_with_ranges.internal_ip_address, 'internal_port_range': self.new_port_forwarding_with_ranges.internal_port_range, 'internal_port_id': self.new_port_forwarding_with_ranges.internal_port_id, 'protocol': self.new_port_forwarding_with_ranges.protocol, 'description': self.new_port_forwarding_with_ranges.description})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)