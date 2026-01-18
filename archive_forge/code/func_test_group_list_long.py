from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_group_list_long(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'domain': None, 'user': None}
    self.groups_mock.list.assert_called_with(**kwargs)
    columns = self.columns + ('Domain ID', 'Description')
    datalist = ((self.group.id, self.group.name, self.group.domain_id, self.group.description),)
    self.assertEqual(columns, columns)
    self.assertEqual(datalist, tuple(data))