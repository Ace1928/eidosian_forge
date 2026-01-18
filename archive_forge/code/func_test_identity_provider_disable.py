import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_identity_provider_disable(self):
    """Disable Identity Provider

        Set Identity Provider's ``enabled`` attribute to False.
        """

    def prepare(self):
        """Prepare fake return objects before the test is executed"""
        updated_idp = copy.deepcopy(identity_fakes.IDENTITY_PROVIDER)
        updated_idp['enabled'] = False
        resources = fakes.FakeResource(None, updated_idp, loaded=True)
        self.identity_providers_mock.update.return_value = resources
    prepare(self)
    arglist = ['--disable', identity_fakes.idp_id, '--remote-id', identity_fakes.idp_remote_ids[0], '--remote-id', identity_fakes.idp_remote_ids[1]]
    verifylist = [('identity_provider', identity_fakes.idp_id), ('description', None), ('enable', False), ('disable', True), ('remote_id', identity_fakes.idp_remote_ids)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.identity_providers_mock.update.assert_called_with(identity_fakes.idp_id, enabled=False, remote_ids=identity_fakes.idp_remote_ids)