from openstack.tests.functional import base
def test_create_list_security_groups_operator(self):
    if not self.operator_cloud:
        self.skipTest('Operator cloud is required for this test')
    sg1 = self.user_cloud.create_security_group(name='sg1', description='sg1')
    self.addCleanup(self.user_cloud.delete_security_group, sg1['id'])
    sg2 = self.operator_cloud.create_security_group(name='sg2', description='sg2')
    self.addCleanup(self.operator_cloud.delete_security_group, sg2['id'])
    if self.user_cloud.has_service('network'):
        sg_list = self.operator_cloud.list_security_groups()
        self.assertIn(sg1['id'], [sg['id'] for sg in sg_list])
        sg_list = self.operator_cloud.list_security_groups(filters={'tenant_id': self.user_cloud.current_project_id})
        self.assertIn(sg1['id'], [sg['id'] for sg in sg_list])
        self.assertNotIn(sg2['id'], [sg['id'] for sg in sg_list])
    else:
        sg_list = self.operator_cloud.list_security_groups()
        self.assertIn(sg2['id'], [sg['id'] for sg in sg_list])
        self.assertNotIn(sg1['id'], [sg['id'] for sg in sg_list])
        sg_list = self.operator_cloud.list_security_groups(filters={'all_tenants': 1})
        self.assertIn(sg1['id'], [sg['id'] for sg in sg_list])