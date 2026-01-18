from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_backup_restore_with_volume(self):
    self.volume_sdk_client.find_volume.side_effect = exceptions.CommandError()
    arglist = [self.backup.id, self.backup.volume_id]
    verifylist = [('backup', self.backup.id), ('volume', self.backup.volume_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.volume_sdk_client.restore_backup.assert_called_with(self.backup.id, volume_id=None, name=self.backup.volume_id)
    self.assertIsNotNone(result)