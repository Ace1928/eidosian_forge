from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_group_set_name_and_description(self):
    arglist = ['--name', 'new_name', '--description', 'new_description', self.group.id]
    verifylist = [('name', 'new_name'), ('description', 'new_description'), ('group', self.group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'name': 'new_name', 'description': 'new_description'}
    self.groups_mock.update.assert_called_once_with(self.group.id, **kwargs)
    self.assertIsNone(result)