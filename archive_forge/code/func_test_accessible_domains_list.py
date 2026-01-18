import copy
from openstackclient.identity.v3 import unscoped_saml
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_accessible_domains_list(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.domains_mock.list.assert_called_with()
    collist = ('ID', 'Enabled', 'Name', 'Description')
    self.assertEqual(collist, columns)
    datalist = ((identity_fakes.domain_id, True, identity_fakes.domain_name, identity_fakes.domain_description),)
    self.assertEqual(datalist, tuple(data))