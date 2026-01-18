import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_endpoint_association_cleanup_when_endpoint_deleted(self):
    url = '/policies/%(policy_id)s/OS-ENDPOINT-POLICY/endpoints/%(endpoint_id)s' % {'policy_id': self.policy['id'], 'endpoint_id': self.endpoint['id']}
    self.put(url)
    self.head(url)
    self.delete('/endpoints/%(endpoint_id)s' % {'endpoint_id': self.endpoint['id']})
    self.head(url, expected_status=http.client.NOT_FOUND)