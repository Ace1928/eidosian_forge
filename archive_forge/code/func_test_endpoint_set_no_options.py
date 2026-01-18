from openstackclient.identity.v3 import endpoint
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_endpoint_set_no_options(self):
    arglist = [self.endpoint.id]
    verifylist = [('endpoint', self.endpoint.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'enabled': None, 'interface': None, 'region': None, 'service': None, 'url': None}
    self.endpoints_mock.update.assert_called_with(self.endpoint.id, **kwargs)
    self.assertIsNone(result)