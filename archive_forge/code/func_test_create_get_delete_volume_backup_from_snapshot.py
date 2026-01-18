from openstack.tests.functional import base
def test_create_get_delete_volume_backup_from_snapshot(self):
    volume = self.user_cloud.create_volume(size=1)
    snapshot = self.user_cloud.create_volume_snapshot(volume['id'])
    self.addCleanup(self.user_cloud.delete_volume, volume['id'])
    self.addCleanup(self.user_cloud.delete_volume_snapshot, snapshot['id'], wait=True)
    backup = self.user_cloud.create_volume_backup(volume_id=volume['id'], snapshot_id=snapshot['id'], wait=True)
    backup = self.user_cloud.get_volume_backup(backup['id'])
    self.assertEqual(backup['snapshot_id'], snapshot['id'])
    self.user_cloud.delete_volume_backup(backup['id'], wait=True)
    self.assertIsNone(self.user_cloud.get_volume_backup(backup['id']))