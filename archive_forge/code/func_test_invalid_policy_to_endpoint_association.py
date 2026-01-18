import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_invalid_policy_to_endpoint_association(self):
    self.assertRaises(exception.InvalidPolicyAssociation, PROVIDERS.endpoint_policy_api.create_policy_association, self.policy[0]['id'])
    self.assertRaises(exception.InvalidPolicyAssociation, PROVIDERS.endpoint_policy_api.create_policy_association, self.policy[0]['id'], endpoint_id=self.endpoint[0]['id'], region_id=self.region[0]['id'])
    self.assertRaises(exception.InvalidPolicyAssociation, PROVIDERS.endpoint_policy_api.create_policy_association, self.policy[0]['id'], endpoint_id=self.endpoint[0]['id'], service_id=self.service[0]['id'])
    self.assertRaises(exception.InvalidPolicyAssociation, PROVIDERS.endpoint_policy_api.create_policy_association, self.policy[0]['id'], region_id=self.region[0]['id'])