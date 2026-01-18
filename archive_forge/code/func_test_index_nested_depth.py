from unittest import mock
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.resources as resources
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_index_nested_depth(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'index', True)
    stack_identity = identifier.HeatIdentifier(self.tenant, 'rubbish', '1')
    req = self._get(stack_identity._tenant_path() + '/resources', {'nested_depth': '99'})
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=[])
    result = self.controller.index(req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id)
    self.assertEqual([], result['resources'])
    mock_call.assert_called_once_with(req.context, ('list_stack_resources', {'stack_identity': stack_identity, 'nested_depth': 99, 'with_detail': False, 'filters': {}}), version='1.25')