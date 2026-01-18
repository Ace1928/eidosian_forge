import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_list_registered_limit_by_filter(self):
    registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
    registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_two['id'], resource_name='snapshot', default_limit=10, id=uuid.uuid4().hex)
    PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1, registered_limit_2])
    hints = driver_hints.Hints()
    hints.add_filter('service_id', self.service_one['id'])
    res = PROVIDERS.unified_limit_api.list_registered_limits(hints)
    self.assertEqual(2, len(res))
    hints = driver_hints.Hints()
    hints.add_filter('region_id', self.region_one['id'])
    res = PROVIDERS.unified_limit_api.list_registered_limits(hints)
    self.assertEqual(1, len(res))
    hints = driver_hints.Hints()
    hints.add_filter('resource_name', 'backup')
    res = PROVIDERS.unified_limit_api.list_registered_limits(hints)
    self.assertEqual(0, len(res))