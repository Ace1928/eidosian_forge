import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_create_identity_provider_authttl_negative(self):
    arglist = ['--authorization-ttl', '-60', identity_fakes.idp_id]
    verifylist = [('identity_provider_id', identity_fakes.idp_id), ('authorization_ttl', -60)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)