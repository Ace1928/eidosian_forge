import copy
import json
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import application_credential
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_application_credential_create_with_options(self):
    name = identity_fakes.app_cred_name
    self.app_creds_mock.create.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.APP_CRED_OPTIONS), loaded=True)
    arglist = [name, '--secret', 'moresecuresecret', '--role', identity_fakes.role_id, '--expiration', identity_fakes.app_cred_expires_str, '--description', 'credential for testing']
    verifylist = [('name', identity_fakes.app_cred_name), ('secret', 'moresecuresecret'), ('role', [identity_fakes.role_id]), ('expiration', identity_fakes.app_cred_expires_str), ('description', 'credential for testing')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'secret': 'moresecuresecret', 'roles': [identity_fakes.role_id], 'expires_at': identity_fakes.app_cred_expires, 'description': 'credential for testing', 'unrestricted': False, 'access_rules': None}
    self.app_creds_mock.create.assert_called_with(name, **kwargs)
    collist = ('access_rules', 'description', 'expires_at', 'id', 'name', 'project_id', 'roles', 'secret', 'unrestricted')
    self.assertEqual(collist, columns)
    datalist = (None, identity_fakes.app_cred_description, identity_fakes.app_cred_expires_str, identity_fakes.app_cred_id, identity_fakes.app_cred_name, identity_fakes.project_id, identity_fakes.role_name, identity_fakes.app_cred_secret, False)
    self.assertEqual(datalist, data)