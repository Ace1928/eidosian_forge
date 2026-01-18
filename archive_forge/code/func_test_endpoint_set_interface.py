from openstackclient.identity.v3 import endpoint
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_endpoint_set_interface(self):
    arglist = ['--interface', 'public', self.endpoint.id]
    verifylist = [('interface', 'public'), ('endpoint', self.endpoint.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'enabled': None, 'interface': 'public', 'url': None, 'region': None, 'service': None}
    self.endpoints_mock.update.assert_called_with(self.endpoint.id, **kwargs)
    self.assertIsNone(result)