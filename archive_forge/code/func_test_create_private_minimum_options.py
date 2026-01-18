from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_private_minimum_options(self):
    arglist = ['--private', '--project', self._network_segment_range.project_id, '--network-type', 'vxlan', '--minimum', str(self._network_segment_range.minimum), '--maximum', str(self._network_segment_range.maximum), self._network_segment_range.name]
    verifylist = [('shared', False), ('project', self._network_segment_range.project_id), ('network_type', 'vxlan'), ('minimum', self._network_segment_range.minimum), ('maximum', self._network_segment_range.maximum), ('name', self._network_segment_range.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_network_segment_range.assert_called_once_with(**{'shared': False, 'project_id': mock.ANY, 'network_type': 'vxlan', 'minimum': self._network_segment_range.minimum, 'maximum': self._network_segment_range.maximum, 'name': self._network_segment_range.name})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)