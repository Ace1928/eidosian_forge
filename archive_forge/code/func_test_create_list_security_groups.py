from openstack.tests.functional import base
def test_create_list_security_groups(self):
    sg1 = self.user_cloud.create_security_group(name='sg1', description='sg1')
    self.addCleanup(self.user_cloud.delete_security_group, sg1['id'])
    if self.user_cloud.has_service('network'):
        sg_list = self.user_cloud.list_security_groups()
        self.assertIn(sg1['id'], [sg['id'] for sg in sg_list])
    else:
        sg_list = self.operator_cloud.list_security_groups()