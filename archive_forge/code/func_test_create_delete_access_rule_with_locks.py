from openstack.tests.functional.shared_file_system import base
def test_create_delete_access_rule_with_locks(self):
    access_rule = self.user_cloud.share.create_access_rule(self.SHARE_ID, access_level='rw', access_type='ip', access_to='203.0.113.10', lock_deletion=True, lock_visibility=True)
    self.user_cloud.share.delete_access_rule(access_rule['id'], self.SHARE_ID, unrestrict=True)