from openstackclient.identity.v3 import endpoint
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_endpoint_list_service(self):
    arglist = ['--service', self.service.id]
    verifylist = [('service', self.service.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'service': self.service.id}
    self.endpoints_mock.list.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    datalist = ((self.endpoint.id, self.endpoint.region, self.service.name, self.service.type, True, self.endpoint.interface, self.endpoint.url),)
    self.assertEqual(datalist, tuple(data))