import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def check_policy(self, policy, policy_ref=None):
    self.assertIsNotNone(policy.id)
    self.assertIn('self', policy.links)
    self.assertIn('/policies/' + policy.id, policy.links['self'])
    if policy_ref:
        self.assertEqual(policy_ref['blob'], policy.blob)
        self.assertEqual(policy_ref['type'], policy.type)
    else:
        self.assertIsNotNone(policy.blob)
        self.assertIsNotNone(policy.type)