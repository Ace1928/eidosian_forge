import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_list_limit_by_limit(self):
    self.config_fixture.config(list_limit=1)
    limit_1 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex, domain_id=None)
    limit_2 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_two['id'], resource_name='snapshot', resource_limit=5, id=uuid.uuid4().hex, domain_id=None)
    PROVIDERS.unified_limit_api.create_limits([limit_1, limit_2])
    hints = driver_hints.Hints()
    limits = PROVIDERS.unified_limit_api.list_limits(hints=hints)
    self.assertEqual(1, len(limits))
    if limits[0]['id'] == limit_1['id']:
        self.assertDictEqual(limit_1, limits[0])
    else:
        self.assertDictEqual(limit_2, limits[0])