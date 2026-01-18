from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor_profile
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
def test_multi_network_flavor_profiles_delete(self):
    arglist = []
    for a in self._network_flavor_profiles:
        arglist.append(a.id)
    verifylist = [('flavor_profile', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for a in self._network_flavor_profiles:
        calls.append(mock.call(a))
    self.network_client.delete_service_profile.assert_has_calls(calls)
    self.assertIsNone(result)