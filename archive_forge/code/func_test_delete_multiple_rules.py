from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_delete_multiple_rules(self):
    arglist = []
    for rule in self.rule_list:
        arglist.append(rule.id)
    verifylist = [('meter_rule_id', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for rule in self.rule_list:
        calls.append(call(rule))
    self.network_client.delete_metering_label_rule.assert_has_calls(calls)
    self.assertIsNone(result)