from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.identity.v3 import credential
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
def test_credential_multi_delete(self):
    arglist = []
    for c in self.credentials:
        arglist.append(c.id)
    verifylist = [('credential', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for c in self.credentials:
        calls.append(call(c.id))
    self.credentials_mock.delete.assert_has_calls(calls)
    self.assertIsNone(result)