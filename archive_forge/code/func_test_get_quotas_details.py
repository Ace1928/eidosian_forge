from openstack.tests.functional import base
def test_get_quotas_details(self):
    if not self.operator_cloud:
        self.skipTest('Operator cloud is required for this test')
    if not self.operator_cloud.has_service('network'):
        self.skipTest('network service not supported by cloud')
    quotas = ['floating_ips', 'networks', 'ports', 'rbac_policies', 'routers', 'subnets', 'subnet_pools', 'security_group_rules', 'security_groups']
    expected_keys = ['limit', 'used', 'reserved']
    'Test getting details about quota usage'
    quota_details = self.operator_cloud.get_network_quotas('demo', details=True)
    for quota in quotas:
        quota_val = quota_details[quota]
        if quota_val:
            for expected_key in expected_keys:
                self.assertIn(expected_key, quota_val)