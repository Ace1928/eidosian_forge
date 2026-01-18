from unittest.mock import call
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group_snapshot
def test_consistency_group_snapshot_create_no_consistency_group(self):
    arglist = ['--description', self._consistency_group_snapshot.description, self._consistency_group_snapshot.name]
    verifylist = [('description', self._consistency_group_snapshot.description), ('snapshot_name', self._consistency_group_snapshot.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.consistencygroups_mock.get.assert_called_once_with(self._consistency_group_snapshot.name)
    self.cgsnapshots_mock.create.assert_called_once_with(self.consistency_group.id, name=self._consistency_group_snapshot.name, description=self._consistency_group_snapshot.description)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)