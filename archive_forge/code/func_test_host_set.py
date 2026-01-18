from unittest import mock
from openstackclient.compute.v2 import host
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils as tests_utils
def test_host_set(self, h_mock):
    h_mock.return_value = self.host
    h_mock.update.return_value = None
    arglist = ['--enable', '--disable-maintenance', self.host['host']]
    verifylist = [('enable', True), ('enable_maintenance', False), ('host', self.host['host'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertIsNone(result)
    h_mock.assert_called_with(self.host['host'], status='enable', maintenance_mode='disable')