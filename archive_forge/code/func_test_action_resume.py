import json
from unittest import mock
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.actions as actions
from heat.common import identifier
from heat.common import policy
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_action_resume(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'resume', True)
    stack_identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '1')
    body = {'resume': None}
    req = self._post(stack_identity._tenant_path() + '/actions', data=json.dumps(body))
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=None)
    result = self.controller.action(req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id, body=body)
    self.assertIsNone(result)
    mock_call.assert_called_once_with(req.context, ('stack_resume', {'stack_identity': stack_identity}))