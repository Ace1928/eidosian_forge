from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.identity.v3 import credential
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
def test_credential_list_with_options(self):
    arglist = ['--user', self.credential.user_id, '--type', self.credential.type]
    verifylist = [('user', self.credential.user_id), ('type', self.credential.type)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'user_id': self.user.id, 'type': self.credential.type}
    self.users_mock.get.assert_called_with(self.credential.user_id)
    self.credentials_mock.list.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, tuple(data))