import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_identity_provider_set_authttl_zero(self):

    def prepare(self):
        """Prepare fake return objects before the test is executed"""
        updated_idp = copy.deepcopy(identity_fakes.IDENTITY_PROVIDER)
        updated_idp['authorization_ttl'] = 0
        resources = fakes.FakeResource(None, updated_idp, loaded=True)
        self.identity_providers_mock.update.return_value = resources
    prepare(self)
    arglist = ['--authorization-ttl', '0', identity_fakes.idp_id]
    verifylist = [('identity_provider', identity_fakes.idp_id), ('enable', False), ('disable', False), ('remote_id', None), ('authorization_ttl', 0)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.identity_providers_mock.update.assert_called_with(identity_fakes.idp_id, authorization_ttl=0)