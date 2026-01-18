from openstackclient.identity.v3 import domain
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_domain_list_no_options(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.domains_mock.list.assert_called_with()
    collist = ('ID', 'Name', 'Enabled', 'Description')
    self.assertEqual(collist, columns)
    datalist = ((self.domain.id, self.domain.name, True, self.domain.description),)
    self.assertEqual(datalist, tuple(data))