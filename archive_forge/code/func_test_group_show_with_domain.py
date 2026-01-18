from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_group_show_with_domain(self):
    get_mock_result = [exceptions.CommandError, self.group]
    self.groups_mock.get = mock.Mock(side_effect=get_mock_result)
    arglist = ['--domain', self.domain.id, self.group.id]
    verifylist = [('domain', self.domain.id), ('group', self.group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.groups_mock.get.assert_any_call(self.group.id, domain_id=self.domain.id)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)