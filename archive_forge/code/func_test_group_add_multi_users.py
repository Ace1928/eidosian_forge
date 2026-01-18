from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_group_add_multi_users(self):
    arglist = [self._group.name, self.users[0].name, self.users[1].name]
    verifylist = [('group', self._group.name), ('user', [self.users[0].name, self.users[1].name])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = [call(self.users[0].id, self._group.id), call(self.users[1].id, self._group.id)]
    self.users_mock.add_to_group.assert_has_calls(calls)
    self.assertIsNone(result)