from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.identity.v3 import credential
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
def test_credential_set_missing_type(self):
    arglist = ['--user', self.credential.user_id, '--data', self.credential.blob, self.credential.id]
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, [])