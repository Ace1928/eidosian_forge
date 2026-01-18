from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet_pool
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_create_no_pool_prefix(self):
    """Make sure --pool-prefix is a required argument"""
    arglist = [self._subnet_pool.name]
    verifylist = [('name', self._subnet_pool.name)]
    self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)