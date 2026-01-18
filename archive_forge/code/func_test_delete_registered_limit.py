import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_delete_registered_limit(self):
    registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
    registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='snapshot', default_limit=5, id=uuid.uuid4().hex)
    PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1, registered_limit_2])
    PROVIDERS.unified_limit_api.delete_registered_limit(registered_limit_1['id'])
    self.assertRaises(exception.RegisteredLimitNotFound, PROVIDERS.unified_limit_api.get_registered_limit, registered_limit_1['id'])
    reg_limits = PROVIDERS.unified_limit_api.list_registered_limits()
    self.assertEqual(1, len(reg_limits))
    self.assertEqual(registered_limit_2['id'], reg_limits[0]['id'])