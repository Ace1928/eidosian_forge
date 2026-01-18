import copy
from openstackclient.identity.v3 import token
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_authorize_request_tokens(self):
    arglist = ['--request-key', identity_fakes.request_token_id, '--role', identity_fakes.role_name]
    verifylist = [('request_key', identity_fakes.request_token_id), ('role', [identity_fakes.role_name])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.request_tokens_mock.authorize.assert_called_with(identity_fakes.request_token_id, [identity_fakes.role_id])
    collist = ('oauth_verifier',)
    self.assertEqual(collist, columns)
    datalist = (identity_fakes.oauth_verifier_pin,)
    self.assertEqual(datalist, data)