from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
def test_consistency_group_create_from_snapshot(self):
    arglist = ['--consistency-group-snapshot', self.consistency_group_snapshot.id, '--description', self.new_consistency_group.description, self.new_consistency_group.name]
    verifylist = [('snapshot', self.consistency_group_snapshot.id), ('description', self.new_consistency_group.description), ('name', self.new_consistency_group.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.types_mock.get.assert_not_called()
    self.cgsnapshots_mock.get.assert_called_once_with(self.consistency_group_snapshot.id)
    self.consistencygroups_mock.create_from_src.assert_called_with(self.consistency_group_snapshot.id, None, name=self.new_consistency_group.name, description=self.new_consistency_group.description)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)