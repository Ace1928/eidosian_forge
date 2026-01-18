from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
def test_consistency_group_show(self):
    arglist = [self.consistency_group.id]
    verifylist = [('consistency_group', self.consistency_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.consistencygroups_mock.get.assert_called_once_with(self.consistency_group.id)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)