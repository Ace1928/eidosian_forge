from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_backup_list_with_options(self):
    arglist = ['--long', '--name', self.backups[0].name, '--status', 'error', '--volume', self.volume.id, '--marker', self.backups[0].id, '--all-projects', '--limit', '3']
    verifylist = [('long', True), ('name', self.backups[0].name), ('status', 'error'), ('volume', self.volume.id), ('marker', self.backups[0].id), ('all_projects', True), ('limit', 3)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_sdk_client.find_volume.assert_called_once_with(self.volume.id, ignore_missing=False)
    self.volume_sdk_client.find_backup.assert_called_once_with(self.backups[0].id, ignore_missing=False)
    self.volume_sdk_client.backups.assert_called_with(name=self.backups[0].name, status='error', volume_id=self.volume.id, all_tenants=True, marker=self.backups[0].id, limit=3)
    self.assertEqual(self.columns_long, columns)
    self.assertCountEqual(self.data_long, list(data))