from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_tunnel_with_physical_network(self):
    arglist = ['--shared', '--network-type', 'vxlan', '--physical-network', self._network_segment_range.physical_network, '--minimum', str(self._network_segment_range.minimum), '--maximum', str(self._network_segment_range.maximum), self._network_segment_range.name]
    verifylist = [('shared', True), ('network_type', 'vxlan'), ('physical_network', self._network_segment_range.physical_network), ('minimum', self._network_segment_range.minimum), ('maximum', self._network_segment_range.maximum), ('name', self._network_segment_range.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)