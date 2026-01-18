from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_group_add_user(self):
    arglist = [self._group.name, self.users[0].name]
    verifylist = [('group', self._group.name), ('user', [self.users[0].name])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.users_mock.add_to_group.assert_called_once_with(self.users[0].id, self._group.id)
    self.assertIsNone(result)