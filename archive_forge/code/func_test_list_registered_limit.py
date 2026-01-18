import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_registered_limit(self):
    r = self.get('/registered_limits', expected_status=http.client.OK)
    self.assertEqual([], r.result.get('registered_limits'))
    ref1 = unit.new_registered_limit_ref(service_id=self.service_id, resource_name='test_resource', region_id=self.region_id)
    ref2 = unit.new_registered_limit_ref(service_id=self.service_id2, resource_name='test_resource', region_id=self.region_id2)
    r = self.post('/registered_limits', body={'registered_limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    id1 = r.result['registered_limits'][0]['id']
    r = self.get('/registered_limits', expected_status=http.client.OK)
    registered_limits = r.result['registered_limits']
    self.assertEqual(len(registered_limits), 2)
    for key in ['service_id', 'region_id', 'resource_name', 'default_limit']:
        if registered_limits[0]['id'] == id1:
            self.assertEqual(registered_limits[0][key], ref1[key])
            self.assertEqual(registered_limits[1][key], ref2[key])
            break
        self.assertEqual(registered_limits[1][key], ref1[key])
        self.assertEqual(registered_limits[0][key], ref2[key])
    r = self.get('/registered_limits?service_id=%s' % self.service_id, expected_status=http.client.OK)
    registered_limits = r.result['registered_limits']
    self.assertEqual(len(registered_limits), 1)
    for key in ['service_id', 'region_id', 'resource_name', 'default_limit']:
        self.assertEqual(registered_limits[0][key], ref1[key])
    r = self.get('/registered_limits?region_id=%s' % self.region_id2, expected_status=http.client.OK)
    registered_limits = r.result['registered_limits']
    self.assertEqual(len(registered_limits), 1)
    for key in ['service_id', 'region_id', 'resource_name', 'default_limit']:
        self.assertEqual(registered_limits[0][key], ref2[key])
    r = self.get('/registered_limits?resource_name=test_resource', expected_status=http.client.OK)
    registered_limits = r.result['registered_limits']
    self.assertEqual(len(registered_limits), 2)