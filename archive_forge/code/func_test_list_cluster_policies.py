import time
from testtools import content
from openstack.tests.functional import base
def test_list_cluster_policies(self):
    policy_name = 'example_policy'
    spec = {'properties': {'adjustment': {'min_step': 1, 'number': 1, 'type': 'CHANGE_IN_CAPACITY'}, 'event': 'CLUSTER_SCALE_IN'}, 'type': 'senlin.policy.scaling', 'version': '1.0'}
    self.addDetail('policy', content.text_content(policy_name))
    policy = self.user_cloud.create_cluster_policy(name=policy_name, spec=spec)
    self.addCleanup(self.cleanup_policy, policy['id'])
    policy_get = self.user_cloud.get_cluster_policy_by_id(policy['id'])
    policies = self.user_cloud.list_cluster_policies()
    policies[0]['created_at'] = 'ignore'
    policy_get['created_at'] = 'ignore'
    self.assertEqual(policies, [policy_get])