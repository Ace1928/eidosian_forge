import copy
from openstackclient.identity.v3 import service_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as service_fakes
def test_create_service_provider_required_options_only(self):
    arglist = ['--auth-url', service_fakes.sp_auth_url, '--service-provider-url', service_fakes.service_provider_url, service_fakes.sp_id]
    verifylist = [('auth_url', service_fakes.sp_auth_url), ('service_provider_url', service_fakes.service_provider_url), ('service_provider_id', service_fakes.sp_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'enabled': True, 'description': None, 'auth_url': service_fakes.sp_auth_url, 'sp_url': service_fakes.service_provider_url}
    self.service_providers_mock.create.assert_called_with(id=service_fakes.sp_id, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)