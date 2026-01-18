from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor_profile
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
def test_create_disable(self):
    arglist = ['--disable', '--driver', self.new_flavor_profile.driver]
    verifylist = [('disable', True), ('driver', self.new_flavor_profile.driver)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_service_profile.assert_called_once_with(**{'enabled': False, 'driver': self.new_flavor_profile.driver})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)