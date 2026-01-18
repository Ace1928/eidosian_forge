import json
from unittest import mock
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.software_configs as software_configs
from heat.common import exception as heat_exc
from heat.common import policy
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
@mock.patch.object(policy.Enforcer, 'enforce')
def test_delete_not_found(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'delete')
    config_id = 'a45559cd-8736-4375-bc39-d6a7bb62ade2'
    req = self._delete('/software_configs/%s' % config_id)
    error = heat_exc.NotFound('Not found %s' % config_id)
    with mock.patch.object(self.controller.rpc_client, 'delete_software_config', side_effect=tools.to_remote_error(error)):
        resp = tools.request_with_middleware(fault.FaultWrapper, self.controller.delete, req, config_id=config_id, tenant_id=self.tenant)
        self.assertEqual(404, resp.json['code'])
        self.assertEqual('NotFound', resp.json['error']['type'])