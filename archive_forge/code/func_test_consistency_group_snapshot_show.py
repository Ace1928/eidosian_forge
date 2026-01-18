from unittest.mock import call
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group_snapshot
def test_consistency_group_snapshot_show(self):
    arglist = [self._consistency_group_snapshot.id]
    verifylist = [('consistency_group_snapshot', self._consistency_group_snapshot.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.cgsnapshots_mock.get.assert_called_once_with(self._consistency_group_snapshot.id)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)