import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_multiple_endpoint_project_associations(self):

    def _create_an_endpoint():
        endpoint_ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id)
        r = self.post('/endpoints', body={'endpoint': endpoint_ref})
        return r.result['endpoint']['id']
    endpoint_id1 = _create_an_endpoint()
    endpoint_id2 = _create_an_endpoint()
    _create_an_endpoint()
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': endpoint_id1})
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': endpoint_id2})
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
    r = self.post('/auth/tokens', body=auth_data)
    self.assertValidProjectScopedTokenResponse(r, require_catalog=True, endpoint_filter=True, ep_filter_assoc=2)