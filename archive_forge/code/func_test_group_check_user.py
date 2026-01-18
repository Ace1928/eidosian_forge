from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_group_check_user(self):
    arglist = [self.group.name, self.user.name]
    verifylist = [('group', self.group.name), ('user', self.user.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.users_mock.check_in_group.assert_called_once_with(self.user.id, self.group.id)
    self.assertIsNone(result)