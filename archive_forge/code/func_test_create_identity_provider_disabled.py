import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_create_identity_provider_disabled(self):
    IDENTITY_PROVIDER = copy.deepcopy(identity_fakes.IDENTITY_PROVIDER)
    IDENTITY_PROVIDER['enabled'] = False
    IDENTITY_PROVIDER['description'] = None
    resource = fakes.FakeResource(None, IDENTITY_PROVIDER, loaded=True)
    self.identity_providers_mock.create.return_value = resource
    arglist = ['--disable', identity_fakes.idp_id]
    verifylist = [('identity_provider_id', identity_fakes.idp_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'remote_ids': None, 'enabled': False, 'description': None, 'domain_id': None}
    self.identity_providers_mock.create.assert_called_with(id=identity_fakes.idp_id, **kwargs)
    self.assertEqual(self.columns, columns)
    datalist = (None, identity_fakes.domain_id, False, identity_fakes.idp_id, identity_fakes.formatted_idp_remote_ids)
    self.assertCountEqual(datalist, data)