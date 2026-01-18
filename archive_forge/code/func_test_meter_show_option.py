from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_meter_show_option(self):
    arglist = [self.new_meter.name]
    verifylist = [('meter', self.new_meter.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.find_metering_label.assert_called_with(self.new_meter.name, ignore_missing=False)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)