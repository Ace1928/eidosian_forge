from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_routers_delete(self):
    arglist = []
    verifylist = []
    for r in self._routers:
        arglist.append(r.name)
    verifylist = [('router', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for r in self._routers:
        calls.append(call(r))
    self.network_client.delete_router.assert_has_calls(calls)
    self.assertIsNone(result)