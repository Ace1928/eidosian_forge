import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_limit_return_count(self):
    ref1 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
    r = self.post('/limits', body={'limits': [ref1]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    limits = r.result['limits']
    self.assertEqual(1, len(limits))
    ref2 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id2, resource_name='snapshot')
    ref3 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='backup')
    r = self.post('/limits', body={'limits': [ref2, ref3]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    limits = r.result['limits']
    self.assertEqual(2, len(limits))