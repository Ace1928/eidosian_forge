import copy
from openstackclient.identity.v3 import service_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as service_fakes
def test_create_service_provider_disabled(self):
    service_provider = copy.deepcopy(service_fakes.SERVICE_PROVIDER)
    service_provider['enabled'] = False
    service_provider['description'] = None
    resource = fakes.FakeResource(None, service_provider, loaded=True)
    self.service_providers_mock.create.return_value = resource
    arglist = ['--auth-url', service_fakes.sp_auth_url, '--service-provider-url', service_fakes.service_provider_url, '--disable', service_fakes.sp_id]
    verifylist = [('auth_url', service_fakes.sp_auth_url), ('service_provider_url', service_fakes.service_provider_url), ('service_provider_id', service_fakes.sp_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'auth_url': service_fakes.sp_auth_url, 'sp_url': service_fakes.service_provider_url, 'enabled': False, 'description': None}
    self.service_providers_mock.create.assert_called_with(id=service_fakes.sp_id, **kwargs)
    self.assertEqual(self.columns, columns)
    datalist = (service_fakes.sp_auth_url, None, False, service_fakes.sp_id, service_fakes.service_provider_url)
    self.assertEqual(datalist, data)