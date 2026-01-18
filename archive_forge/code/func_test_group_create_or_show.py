from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_group_create_or_show(self):
    self.groups_mock.create.side_effect = ks_exc.Conflict()
    arglist = ['--or-show', self.group.name]
    verifylist = [('or_show', True), ('name', self.group.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.groups_mock.get.assert_called_once_with(self.group.name)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)