import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_create_identity_provider_remote_id(self):
    arglist = [identity_fakes.idp_id, '--remote-id', identity_fakes.idp_remote_ids[0]]
    verifylist = [('identity_provider_id', identity_fakes.idp_id), ('remote_id', identity_fakes.idp_remote_ids[:1])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'remote_ids': identity_fakes.idp_remote_ids[:1], 'description': None, 'domain_id': None, 'enabled': True}
    self.identity_providers_mock.create.assert_called_with(id=identity_fakes.idp_id, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)