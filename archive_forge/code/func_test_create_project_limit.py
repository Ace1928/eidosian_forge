import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_create_project_limit(self):
    limit_1 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex, description='test description', domain_id=None)
    limits = PROVIDERS.unified_limit_api.create_limits([limit_1])
    self.assertDictEqual(limit_1, limits[0])
    limit_2 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_two['id'], resource_name='snapshot', resource_limit=5, id=uuid.uuid4().hex, domain_id=None)
    limit_3 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_two['id'], resource_name='backup', resource_limit=5, id=uuid.uuid4().hex, domain_id=None)
    limits = PROVIDERS.unified_limit_api.create_limits([limit_2, limit_3])
    for limit in limits:
        if limit['id'] == limit_2['id']:
            self.assertDictEqual(limit_2, limit)
        if limit['id'] == limit_3['id']:
            self.assertDictEqual(limit_3, limit)