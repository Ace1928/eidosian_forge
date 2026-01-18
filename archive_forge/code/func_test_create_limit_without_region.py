import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_limit_without_region(self):
    ref = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id2, resource_name='snapshot')
    r = self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    limits = r.result['limits']
    self.assertIsNotNone(limits[0]['id'])
    self.assertIsNotNone(limits[0]['project_id'])
    for key in ['service_id', 'resource_name', 'resource_limit']:
        self.assertEqual(limits[0][key], ref[key])
    self.assertIsNone(limits[0].get('region_id'))