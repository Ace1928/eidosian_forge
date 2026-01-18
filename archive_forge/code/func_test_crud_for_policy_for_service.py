import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_crud_for_policy_for_service(self):
    """PUT, HEAD and DELETE for service endpoint policy."""
    url = '/policies/%(policy_id)s/OS-ENDPOINT-POLICY/services/%(service_id)s' % {'policy_id': self.policy['id'], 'service_id': self.service['id']}
    self._crud_test(url)