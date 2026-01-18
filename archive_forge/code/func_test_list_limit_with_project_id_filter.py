import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_limit_with_project_id_filter(self):
    self.config_fixture.config(group='oslo_policy', enforce_scope=True)
    ref1 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
    ref2 = unit.new_limit_ref(project_id=self.project_2_id, service_id=self.service_id2, resource_name='snapshot')
    self.post('/limits', body={'limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    r = self.get('/limits', expected_status=http.client.OK)
    limits = r.result['limits']
    self.assertEqual(1, len(limits))
    self.assertEqual(self.project_id, limits[0]['project_id'])
    r = self.get('/limits', expected_status=http.client.OK, auth=self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project_2_id))
    limits = r.result['limits']
    self.assertEqual(1, len(limits))
    self.assertEqual(self.project_2_id, limits[0]['project_id'])
    r = self.get('/limits?project_id=%s' % self.project_id, expected_status=http.client.OK)
    limits = r.result['limits']
    self.assertEqual(1, len(limits))
    self.assertEqual(self.project_id, limits[0]['project_id'])
    r = self.get('/limits?project_id=%s' % self.project_id, expected_status=http.client.OK, token=self.system_admin_token)
    limits = r.result['limits']
    self.assertEqual(1, len(limits))
    self.assertEqual(self.project_id, limits[0]['project_id'])