from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
def test_delete_multiple_consistency_groups(self):
    arglist = []
    for b in self.consistency_groups:
        arglist.append(b.id)
    verifylist = [('consistency_groups', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for b in self.consistency_groups:
        calls.append(call(b.id, False))
    self.consistencygroups_mock.delete.assert_has_calls(calls)
    self.assertIsNone(result)