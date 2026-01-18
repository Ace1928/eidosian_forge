from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_multi_ports_delete(self):
    arglist = []
    verifylist = []
    for p in self._ports:
        arglist.append(p.name)
    verifylist = [('port', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for p in self._ports:
        calls.append(call(p))
    self.network_client.delete_port.assert_has_calls(calls)
    self.assertIsNone(result)