from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.identity.v3 import credential
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
def test_credential_multi_delete_with_exception(self):
    arglist = [self.credentials[0].id, 'unexist_credential']
    verifylist = [('credential', [self.credentials[0].id, 'unexist_credential'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    delete_mock_result = [None, exceptions.CommandError]
    self.credentials_mock.delete = mock.Mock(side_effect=delete_mock_result)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 credential failed to delete.', str(e))
    self.credentials_mock.delete.assert_any_call(self.credentials[0].id)
    self.credentials_mock.delete.assert_any_call('unexist_credential')