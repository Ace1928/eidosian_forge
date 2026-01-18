from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_create_default_options(self, net_mock):
    net_mock.return_value = self._network
    arglist = ['--subnet', self._network['cidr'], self._network['label']]
    verifylist = [('subnet', self._network['cidr']), ('name', self._network['label'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    net_mock.assert_called_once_with(**{'subnet': self._network['cidr'], 'name': self._network['label']})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)