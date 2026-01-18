from unittest import mock
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.events as events
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_index_stack_nonexist(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'index', True)
    stack_identity = identifier.HeatIdentifier(self.tenant, 'wibble', '6')
    req = self._get(stack_identity._tenant_path() + '/events')
    kwargs = {'stack_identity': stack_identity, 'nested_depth': None, 'limit': None, 'sort_keys': None, 'marker': None, 'sort_dir': None, 'filters': None}
    error = heat_exc.EntityNotFound(entity='Stack', name='a')
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', side_effect=tools.to_remote_error(error))
    resp = tools.request_with_middleware(fault.FaultWrapper, self.controller.index, req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id)
    self.assertEqual(404, resp.json['code'])
    self.assertEqual('EntityNotFound', resp.json['error']['type'])
    mock_call.assert_called_once_with(req.context, ('list_events', kwargs), version='1.31')