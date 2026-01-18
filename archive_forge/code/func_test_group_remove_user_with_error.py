from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
@mock.patch.object(group.LOG, 'error')
def test_group_remove_user_with_error(self, mock_error):
    self.users_mock.remove_from_group.side_effect = [exceptions.CommandError(), None]
    arglist = [self._group.id, self.users[0].id, self.users[1].id]
    verifylist = [('group', self._group.id), ('user', [self.users[0].id, self.users[1].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        msg = '1 of 2 users not removed from group %s.' % self._group.id
        self.assertEqual(msg, str(e))
    msg = '%(user)s not removed from group %(group)s: ' % {'user': self.users[0].id, 'group': self._group.id}
    mock_error.assert_called_once_with(msg)