from openstackclient.identity.v3 import domain
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_domain_show(self):
    arglist = [self.domain.id]
    verifylist = [('domain', self.domain.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.app.client_manager.identity.tokens.get_token_data.return_value = {'token': {'project': {'domain': {'id': 'd1', 'name': 'd1'}}}}
    columns, data = self.cmd.take_action(parsed_args)
    self.domains_mock.get.assert_called_with(self.domain.id)
    collist = ('description', 'enabled', 'id', 'name', 'tags')
    self.assertEqual(collist, columns)
    datalist = (self.domain.description, True, self.domain.id, self.domain.name, self.domain.tags)
    self.assertEqual(datalist, data)