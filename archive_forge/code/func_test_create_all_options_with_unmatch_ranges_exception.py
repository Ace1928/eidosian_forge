from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip_port_forwarding
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_all_options_with_unmatch_ranges_exception(self):
    internal_range = '80:90'
    external_range = '8080:8100'
    arglist = ['--port', self.new_port_forwarding_with_ranges.internal_port_id, '--internal-protocol-port', internal_range, '--external-protocol-port', external_range, '--protocol', self.new_port_forwarding_with_ranges.protocol, self.new_port_forwarding_with_ranges.floatingip_id, '--internal-ip-address', self.new_port_forwarding_with_ranges.internal_ip_address, '--description', self.new_port_forwarding_with_ranges.description]
    verifylist = [('port', self.new_port_forwarding_with_ranges.internal_port_id), ('internal_protocol_port', internal_range), ('external_protocol_port', external_range), ('protocol', self.new_port_forwarding_with_ranges.protocol), ('floating_ip', self.new_port_forwarding_with_ranges.floatingip_id), ('internal_ip_address', self.new_port_forwarding_with_ranges.internal_ip_address), ('description', self.new_port_forwarding_with_ranges.description)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    msg = 'The relation between internal and external ports does not match the pattern 1:N and N:N'
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual(msg, str(e))
        self.network_client.create_floating_ip_port_forwarding.assert_not_called()