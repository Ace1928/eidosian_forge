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
def test_delete_trunk_multiple(self):
    arglist = []
    verifylist = []
    for t in self.new_trunks:
        arglist.append(t['name'])
    verifylist = [('trunk', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for t in self.new_trunks:
        calls.append(call(t.id))
    self.network_client.delete_trunk.assert_has_calls(calls)
    self.assertIsNone(result)