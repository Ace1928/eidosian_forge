from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_delete_one_meter(self):
    arglist = [self.meter_list[0].name]
    verifylist = [('meter', [self.meter_list[0].name])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.delete_metering_label.assert_called_once_with(self.meter_list[0])
    self.assertIsNone(result)