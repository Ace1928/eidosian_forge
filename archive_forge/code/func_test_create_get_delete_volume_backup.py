from openstack.tests.functional import base
def test_create_get_delete_volume_backup(self):
    volume = self.user_cloud.create_volume(display_name=self.getUniqueString(), size=1)
    self.addCleanup(self.user_cloud.delete_volume, volume['id'])
    backup_name_1 = self.getUniqueString()
    backup_desc_1 = self.getUniqueString()
    backup = self.user_cloud.create_volume_backup(volume_id=volume['id'], name=backup_name_1, description=backup_desc_1, wait=True)
    self.assertEqual(backup_name_1, backup['name'])
    backup = self.user_cloud.get_volume_backup(backup['id'])
    self.assertEqual('available', backup['status'])
    self.assertEqual(backup_desc_1, backup['description'])
    self.user_cloud.delete_volume_backup(backup['id'], wait=True)
    self.assertIsNone(self.user_cloud.get_volume_backup(backup['id']))