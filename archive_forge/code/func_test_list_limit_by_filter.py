import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_list_limit_by_filter(self):
    limit_1 = unit.new_limit_ref(project_id=self.project_bar['id'], service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', resource_limit=10, id=uuid.uuid4().hex, domain_id=None)
    limit_2 = unit.new_limit_ref(project_id=self.project_baz['id'], service_id=self.service_one['id'], region_id=self.region_two['id'], resource_name='snapshot', resource_limit=10, id=uuid.uuid4().hex, domain_id=None)
    limit_3 = unit.new_limit_ref(project_id=None, service_id=self.service_one['id'], region_id=self.region_two['id'], resource_name='snapshot', resource_limit=10, id=uuid.uuid4().hex, domain_id=self.domain_default['id'])
    PROVIDERS.unified_limit_api.create_limits([limit_1, limit_2, limit_3])
    hints = driver_hints.Hints()
    hints.add_filter('service_id', self.service_one['id'])
    res = PROVIDERS.unified_limit_api.list_limits(hints)
    self.assertEqual(3, len(res))
    hints = driver_hints.Hints()
    hints.add_filter('region_id', self.region_one['id'])
    res = PROVIDERS.unified_limit_api.list_limits(hints)
    self.assertEqual(1, len(res))
    self.assertDictEqual(limit_1, res[0])
    hints = driver_hints.Hints()
    hints.add_filter('resource_name', 'backup')
    res = PROVIDERS.unified_limit_api.list_limits(hints)
    self.assertEqual(0, len(res))
    hints = driver_hints.Hints()
    hints.add_filter('project_id', self.project_bar['id'])
    res = PROVIDERS.unified_limit_api.list_limits(hints)
    self.assertEqual(1, len(res))
    hints = driver_hints.Hints()
    hints.add_filter('domain_id', self.domain_default['id'])
    res = PROVIDERS.unified_limit_api.list_limits(hints)
    self.assertEqual(1, len(res))