from libcloud.backup.base import BackupTarget, BackupTargetType
def test_list_targets_response(self):
    targets = self.driver.list_targets()
    self.assertTrue(isinstance(targets, list))
    for target in targets:
        self.assertTrue(isinstance(target, BackupTarget))