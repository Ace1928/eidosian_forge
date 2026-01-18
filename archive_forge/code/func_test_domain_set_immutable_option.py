from openstackclient.identity.v3 import domain
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_domain_set_immutable_option(self):
    arglist = ['--immutable', self.domain.id]
    verifylist = [('immutable', True), ('domain', self.domain.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'options': {'immutable': True}}
    self.domains_mock.update.assert_called_with(self.domain.id, **kwargs)
    self.assertIsNone(result)