from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_no_qos_policy_option(self):
    arglist = ['--no-qos-policy', self.floating_ip.id]
    verifylist = [('no_qos_policy', True), ('floating_ip', self.floating_ip.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    attrs = {'qos_policy_id': None}
    self.network_client.find_ip.assert_called_once_with(self.floating_ip.id, ignore_missing=False)
    self.network_client.update_ip.assert_called_once_with(self.floating_ip, **attrs)