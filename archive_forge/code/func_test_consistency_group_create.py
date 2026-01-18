from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
def test_consistency_group_create(self):
    arglist = ['--volume-type', self.volume_type.id, '--description', self.new_consistency_group.description, '--availability-zone', self.new_consistency_group.availability_zone, self.new_consistency_group.name]
    verifylist = [('volume_type', self.volume_type.id), ('description', self.new_consistency_group.description), ('availability_zone', self.new_consistency_group.availability_zone), ('name', self.new_consistency_group.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.types_mock.get.assert_called_once_with(self.volume_type.id)
    self.consistencygroups_mock.get.assert_not_called()
    self.consistencygroups_mock.create.assert_called_once_with(self.volume_type.id, name=self.new_consistency_group.name, description=self.new_consistency_group.description, availability_zone=self.new_consistency_group.availability_zone)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)