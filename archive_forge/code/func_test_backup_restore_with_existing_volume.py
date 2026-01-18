from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v1 import fakes as volume_fakes
from openstackclient.volume.v1 import volume_backup
def test_backup_restore_with_existing_volume(self):
    arglist = [self.backup.id, self.backup.volume_id]
    verifylist = [('backup', self.backup.id), ('volume', self.backup.volume_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.restores_mock.restore.assert_called_with(self.backup.id, self.backup.volume_id)
    self.assertIsNotNone(result)