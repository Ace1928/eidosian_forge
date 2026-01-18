from openstack.tests.functional import base
def test_get_security_group_by_id(self):
    sg = self.user_cloud.create_security_group(name='sg', description='sg')
    self.addCleanup(self.user_cloud.delete_security_group, sg['id'])
    ret_sg = self.user_cloud.get_security_group_by_id(sg['id'])
    self.assertEqual(sg, ret_sg)