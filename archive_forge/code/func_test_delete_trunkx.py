import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
import testtools
from openstackclient.network.v2 import network_trunk
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_delete_trunkx(self):
    arglist = [self.new_trunks[0].name]
    verifylist = [('trunk', [self.new_trunks[0].name])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.delete_trunk.assert_called_once_with(self.new_trunks[0].id)
    self.assertIsNone(result)