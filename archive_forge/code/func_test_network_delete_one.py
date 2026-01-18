from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_delete_one(self, net_mock):
    net_mock.return_value = mock.Mock(return_value=None)
    arglist = [self._networks[0]['label']]
    verifylist = [('network', [self._networks[0]['label']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    net_mock.assert_called_once_with(self._networks[0]['label'])
    self.assertIsNone(result)