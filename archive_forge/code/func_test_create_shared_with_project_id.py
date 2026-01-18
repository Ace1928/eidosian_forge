from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_shared_with_project_id(self):
    arglist = ['--shared', '--project', self._network_segment_range.project_id, '--network-type', 'vxlan', '--minimum', str(self._network_segment_range.minimum), '--maximum', str(self._network_segment_range.maximum), self._network_segment_range.name]
    verifylist = [('shared', True), ('project', self._network_segment_range.project_id), ('network_type', 'vxlan'), ('minimum', self._network_segment_range.minimum), ('maximum', self._network_segment_range.maximum), ('name', self._network_segment_range.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)