from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_scope
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_address_scopes_delete(self):
    arglist = []
    verifylist = []
    for a in self._address_scopes:
        arglist.append(a.name)
    verifylist = [('address_scope', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for a in self._address_scopes:
        calls.append(call(a))
    self.network_client.delete_address_scope.assert_has_calls(calls)
    self.assertIsNone(result)