from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_wrong_options(self):
    arglist = ['--type', RULE_TYPE_BANDWIDTH_LIMIT, '--min-kbps', '10000', self.new_rule.qos_policy_id]
    verifylist = [('type', RULE_TYPE_BANDWIDTH_LIMIT), ('min_kbps', 10000), ('qos_policy', self.new_rule.qos_policy_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
    except exceptions.CommandError as e:
        msg = 'Failed to create Network QoS rule: "Create" rule command for type "bandwidth-limit" requires arguments: max_kbps'
        self.assertEqual(msg, str(e))