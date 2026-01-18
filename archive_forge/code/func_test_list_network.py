from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_list_network(self):
    arglist = ['--network', self._network.id]
    verifylist = [('long', False), ('network', self._network.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.segments.assert_called_once_with(**{'network_id': self._network.id})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))