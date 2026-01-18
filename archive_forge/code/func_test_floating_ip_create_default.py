from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_floating_ip_create_default(self, fip_mock):
    fip_mock.return_value = self._floating_ip
    arglist = [self._floating_ip['pool']]
    verifylist = [('network', self._floating_ip['pool'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    fip_mock.assert_called_once_with(self._floating_ip['pool'])
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)