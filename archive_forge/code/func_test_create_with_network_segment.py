from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_with_network_segment(self):
    self._network.id = self._subnet.network_id
    arglist = ['--subnet-range', self._subnet.cidr, '--network-segment', self._network_segment.id, '--network', self._subnet.network_id, self._subnet.name]
    verifylist = [('name', self._subnet.name), ('subnet_range', self._subnet.cidr), ('network_segment', self._network_segment.id), ('network', self._subnet.network_id), ('ip_version', self._subnet.ip_version), ('gateway', 'auto')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_subnet.assert_called_once_with(**{'cidr': self._subnet.cidr, 'ip_version': self._subnet.ip_version, 'name': self._subnet.name, 'network_id': self._subnet.network_id, 'segment_id': self._network_segment.id})
    self.assertFalse(self.network_client.set_tags.called)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)