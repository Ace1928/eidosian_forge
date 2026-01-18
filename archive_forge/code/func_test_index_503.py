from unittest import mock
from oslo_messaging import exceptions
import webob.exc
import heat.api.openstack.v1.services as services
from heat.common import policy
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
@mock.patch.object(policy.Enforcer, 'enforce')
def test_index_503(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'index')
    req = self._get('/services')
    with mock.patch.object(self.controller.rpc_client, 'list_services', side_effect=exceptions.MessagingTimeout()):
        self.assertRaises(webob.exc.HTTPServiceUnavailable, self.controller.index, req, tenant_id=self.tenant)