import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_create_identity_provider_authttl_not_int(self):
    arglist = ['--authorization-ttl', 'spam', identity_fakes.idp_id]
    verifylist = []
    self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)