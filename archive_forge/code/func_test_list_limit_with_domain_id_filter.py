import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_limit_with_domain_id_filter(self):
    ref1 = unit.new_limit_ref(domain_id=self.domain_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
    ref2 = unit.new_limit_ref(domain_id=self.domain_2_id, service_id=self.service_id2, resource_name='snapshot')
    self.post('/limits', body={'limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    r = self.get('/limits', expected_status=http.client.OK, auth=self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain_id))
    limits = r.result['limits']
    self.assertEqual(1, len(limits))
    self.assertEqual(self.domain_id, limits[0]['domain_id'])
    r = self.get('/limits', expected_status=http.client.OK, auth=self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain_2_id))
    limits = r.result['limits']
    self.assertEqual(1, len(limits))
    self.assertEqual(self.domain_2_id, limits[0]['domain_id'])
    r = self.get('/limits?domain_id=%s' % self.domain_id, expected_status=http.client.OK)
    limits = r.result['limits']
    self.assertEqual(0, len(limits))
    r = self.get('/limits?domain_id=%s' % self.domain_id, expected_status=http.client.OK, auth=self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], system=True))
    limits = r.result['limits']
    self.assertEqual(1, len(limits))
    self.assertEqual(self.domain_id, limits[0]['domain_id'])