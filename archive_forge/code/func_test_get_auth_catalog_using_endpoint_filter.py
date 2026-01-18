import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_auth_catalog_using_endpoint_filter(self):
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': self.endpoint_id})
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
    token_data = self.post('/auth/tokens', body=auth_data)
    self.assertValidProjectScopedTokenResponse(token_data, require_catalog=True, endpoint_filter=True, ep_filter_assoc=1)
    auth_catalog = self.get('/auth/catalog', token=token_data.headers['X-Subject-Token'])
    self.assertEqual(token_data.result['token']['catalog'], auth_catalog.result['catalog'])