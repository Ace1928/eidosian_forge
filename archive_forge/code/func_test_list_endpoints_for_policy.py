import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_endpoints_for_policy(self):
    """GET & HEAD /policies/%(policy_id}/endpoints."""
    url = '/policies/%(policy_id)s/OS-ENDPOINT-POLICY/endpoints' % {'policy_id': self.policy['id']}
    self.put(url + '/' + self.endpoint['id'])
    r = self.get(url)
    self.assertValidEndpointListResponse(r, ref=self.endpoint)
    self.assertThat(r.result.get('endpoints'), matchers.HasLength(1))
    self.head(url, expected_status=http.client.OK)