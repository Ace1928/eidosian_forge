from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
@mock.patch.object(group.LOG, 'error')
def test_group_add_user_with_error(self, mock_error):
    self.users_mock.add_to_group.side_effect = [exceptions.CommandError(), None]
    arglist = [self._group.name, self.users[0].name, self.users[1].name]
    verifylist = [('group', self._group.name), ('user', [self.users[0].name, self.users[1].name])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        msg = '1 of 2 users not added to group %s.' % self._group.name
        self.assertEqual(msg, str(e))
    msg = '%(user)s not added to group %(group)s: ' % {'user': self.users[0].name, 'group': self._group.name}
    mock_error.assert_called_once_with(msg)