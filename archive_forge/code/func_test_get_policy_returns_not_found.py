import uuid
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_get_policy_returns_not_found(self):
    self.assertRaises(exception.PolicyNotFound, PROVIDERS.policy_api.get_policy, uuid.uuid4().hex)