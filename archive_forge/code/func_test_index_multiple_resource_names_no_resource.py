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
@mock.patch.object(rpc_client.EngineClient, 'call')
def test_index_multiple_resource_names_no_resource(self, mock_call, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'index', True)
    res_name = 'resource3'
    event_id = '42'
    params = {'resource_name': ['resource1', 'resource2']}
    stack_identity = identifier.HeatIdentifier(self.tenant, 'wibble', '6')
    res_identity = identifier.ResourceIdentifier(resource_name=res_name, **stack_identity)
    ev_identity = identifier.EventIdentifier(event_id=event_id, **res_identity)
    req = self._get(stack_identity._tenant_path() + '/events', params=params)
    mock_call.return_value = [{u'stack_name': u'wordpress', u'event_time': u'2012-07-23T13:05:39Z', u'stack_identity': dict(stack_identity), u'resource_name': res_name, u'resource_status_reason': u'state changed', u'event_identity': dict(ev_identity), u'resource_action': u'CREATE', u'resource_status': u'IN_PROGRESS', u'physical_resource_id': None, u'resource_type': u'AWS::EC2::Instance'}]
    self.controller.index(req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id)
    rpc_call_args, _ = mock_call.call_args
    engine_args = rpc_call_args[1][1]
    self.assertEqual(7, len(engine_args))
    self.assertIn('filters', engine_args)
    self.assertIn('resource_name', engine_args['filters'])
    self.assertIn('resource1', engine_args['filters']['resource_name'])
    self.assertIn('resource2', engine_args['filters']['resource_name'])