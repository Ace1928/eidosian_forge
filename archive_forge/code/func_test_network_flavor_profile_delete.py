from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor_profile
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
def test_network_flavor_profile_delete(self):
    arglist = [self._network_flavor_profiles[0].id]
    verifylist = [('flavor_profile', [self._network_flavor_profiles[0].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.find_service_profile.assert_called_once_with(self._network_flavor_profiles[0].id, ignore_missing=False)
    self.network_client.delete_service_profile.assert_called_once_with(self._network_flavor_profiles[0])
    self.assertIsNone(result)