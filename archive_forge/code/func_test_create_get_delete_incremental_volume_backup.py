from openstack.tests.functional import base
def test_create_get_delete_incremental_volume_backup(self):
    volume = self.user_cloud.create_volume(size=1)
    self.addCleanup(self.user_cloud.delete_volume, volume['id'])
    full_backup = self.user_cloud.create_volume_backup(volume_id=volume['id'], wait=True)
    incr_backup = self.user_cloud.create_volume_backup(volume_id=volume['id'], incremental=True, wait=True)
    full_backup = self.user_cloud.get_volume_backup(full_backup['id'])
    incr_backup = self.user_cloud.get_volume_backup(incr_backup['id'])
    self.assertEqual(full_backup['has_dependent_backups'], True)
    self.assertEqual(incr_backup['is_incremental'], True)
    self.user_cloud.delete_volume_backup(incr_backup['id'], wait=True)
    self.user_cloud.delete_volume_backup(full_backup['id'], wait=True)
    self.assertIsNone(self.user_cloud.get_volume_backup(full_backup['id']))
    self.assertIsNone(self.user_cloud.get_volume_backup(incr_backup['id']))