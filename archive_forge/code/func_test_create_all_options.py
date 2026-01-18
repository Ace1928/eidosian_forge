from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_all_options(self):
    arglist = ['--ingress', '--include', self.new_rule.metering_label_id, '--remote-ip-prefix', self.new_rule.remote_ip_prefix]
    verifylist = [('ingress', True), ('include', True), ('meter', self.new_rule.metering_label_id), ('remote_ip_prefix', self.new_rule.remote_ip_prefix)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_metering_label_rule.assert_called_once_with(**{'direction': self.new_rule.direction, 'excluded': self.new_rule.excluded, 'metering_label_id': self.new_rule.metering_label_id, 'remote_ip_prefix': self.new_rule.remote_ip_prefix})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)