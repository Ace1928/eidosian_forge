import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_registered_limit_return_count(self):
    ref1 = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id)
    r = self.post('/registered_limits', body={'registered_limits': [ref1]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    registered_limits = r.result['registered_limits']
    self.assertEqual(1, len(registered_limits))
    ref2 = unit.new_registered_limit_ref(service_id=self.service_id2, region_id=self.region_id2)
    ref3 = unit.new_registered_limit_ref(service_id=self.service_id2)
    r = self.post('/registered_limits', body={'registered_limits': [ref2, ref3]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    registered_limits = r.result['registered_limits']
    self.assertEqual(2, len(registered_limits))