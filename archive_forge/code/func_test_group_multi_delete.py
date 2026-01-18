from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_group_multi_delete(self):
    arglist = []
    verifylist = []
    for g in self.groups:
        arglist.append(g.id)
    verifylist = [('groups', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for g in self.groups:
        calls.append(call(g.id))
    self.groups_mock.delete.assert_has_calls(calls)
    self.assertIsNone(result)