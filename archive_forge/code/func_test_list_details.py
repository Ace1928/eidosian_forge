from openstack.tests.functional import base
def test_list_details(self):
    expected_keys = ['limit', 'used', 'reserved']
    project_id = self.operator_cloud.session.get_project_id()
    quota_details = self.operator_cloud.network.get_quota(project_id, details=True)
    for details in quota_details._body.attributes.values():
        for expected_key in expected_keys:
            self.assertTrue(expected_key in details.keys())