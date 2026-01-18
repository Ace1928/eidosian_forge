import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_delete_association_by_entity(self):
    PROVIDERS.endpoint_policy_api.create_policy_association(self.policy[0]['id'], endpoint_id=self.endpoint[0]['id'])
    PROVIDERS.endpoint_policy_api.delete_association_by_endpoint(self.endpoint[0]['id'])
    self.assertRaises(exception.NotFound, PROVIDERS.endpoint_policy_api.check_policy_association, self.policy[0]['id'], endpoint_id=self.endpoint[0]['id'])
    PROVIDERS.endpoint_policy_api.delete_association_by_endpoint(self.endpoint[0]['id'])
    PROVIDERS.endpoint_policy_api.create_policy_association(self.policy[0]['id'], service_id=self.service[0]['id'], region_id=self.region[0]['id'])
    PROVIDERS.endpoint_policy_api.create_policy_association(self.policy[1]['id'], service_id=self.service[0]['id'], region_id=self.region[1]['id'])
    PROVIDERS.endpoint_policy_api.create_policy_association(self.policy[0]['id'], service_id=self.service[0]['id'])
    PROVIDERS.endpoint_policy_api.delete_association_by_service(self.service[0]['id'])
    self.assertRaises(exception.NotFound, PROVIDERS.endpoint_policy_api.check_policy_association, self.policy[0]['id'], service_id=self.service[0]['id'], region_id=self.region[0]['id'])
    self.assertRaises(exception.NotFound, PROVIDERS.endpoint_policy_api.check_policy_association, self.policy[1]['id'], service_id=self.service[0]['id'], region_id=self.region[1]['id'])
    self.assertRaises(exception.NotFound, PROVIDERS.endpoint_policy_api.check_policy_association, self.policy[0]['id'], service_id=self.service[0]['id'])
    PROVIDERS.endpoint_policy_api.create_policy_association(self.policy[0]['id'], service_id=self.service[0]['id'], region_id=self.region[0]['id'])
    PROVIDERS.endpoint_policy_api.delete_association_by_region(self.region[0]['id'])
    self.assertRaises(exception.NotFound, PROVIDERS.endpoint_policy_api.check_policy_association, self.policy[0]['id'], service_id=self.service[0]['id'], region_id=self.region[0]['id'])
    self.assertRaises(exception.NotFound, PROVIDERS.endpoint_policy_api.check_policy_association, self.policy[0]['id'], service_id=self.service[0]['id'])