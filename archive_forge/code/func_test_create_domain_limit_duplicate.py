import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_create_domain_limit_duplicate(self):
    limit_1 = unit.new_limit_ref(project_id=None, service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex, domain_id=self.domain_default['id'])
    PROVIDERS.unified_limit_api.create_limits([limit_1])
    limit_1 = unit.new_limit_ref(project_id=None, service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex, domain_id=self.domain_default['id'])
    self.assertRaises(exception.Conflict, PROVIDERS.unified_limit_api.create_limits, [limit_1])