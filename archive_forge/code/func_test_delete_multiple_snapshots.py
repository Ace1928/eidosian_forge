from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
def test_delete_multiple_snapshots(self):
    arglist = []
    for s in self.snapshots:
        arglist.append(s.id)
    verifylist = [('snapshots', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for s in self.snapshots:
        calls.append(mock.call(s.id, False))
    self.snapshots_mock.delete.assert_has_calls(calls)
    self.assertIsNone(result)