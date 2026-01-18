from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.identity.v3 import credential
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
def test_credential_set_valid_with_project(self):
    arglist = ['--user', self.credential.user_id, '--type', 'ec2', '--data', self.credential.blob, '--project', self.credential.project_id, self.credential.id]
    parsed_args = self.check_parser(self.cmd, arglist, [])
    result = self.cmd.take_action(parsed_args)
    self.assertIsNone(result)