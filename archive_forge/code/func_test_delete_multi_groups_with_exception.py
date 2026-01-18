from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
@mock.patch.object(utils, 'find_resource')
def test_delete_multi_groups_with_exception(self, find_mock):
    find_mock.side_effect = [self.groups[0], exceptions.CommandError]
    arglist = [self.groups[0].id, 'unexist_group']
    verifylist = [('groups', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 groups failed to delete.', str(e))
    find_mock.assert_any_call(self.groups_mock, self.groups[0].id)
    find_mock.assert_any_call(self.groups_mock, 'unexist_group')
    self.assertEqual(2, find_mock.call_count)
    self.groups_mock.delete.assert_called_once_with(self.groups[0].id)