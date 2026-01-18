from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_label_rule_show_option(self):
    arglist = [self.new_rule.id]
    verifylist = [('meter_rule_id', self.new_rule.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.find_metering_label_rule.assert_called_with(self.new_rule.id, ignore_missing=False)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)