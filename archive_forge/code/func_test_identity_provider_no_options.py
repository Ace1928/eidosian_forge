import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_identity_provider_no_options(self):

    def prepare(self):
        """Prepare fake return objects before the test is executed"""
        resources = fakes.FakeResource(None, copy.deepcopy(identity_fakes.IDENTITY_PROVIDER), loaded=True)
        self.identity_providers_mock.get.return_value = resources
        resources = fakes.FakeResource(None, copy.deepcopy(identity_fakes.IDENTITY_PROVIDER), loaded=True)
        self.identity_providers_mock.update.return_value = resources
    prepare(self)
    arglist = [identity_fakes.idp_id]
    verifylist = [('identity_provider', identity_fakes.idp_id), ('enable', False), ('disable', False), ('remote_id', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)