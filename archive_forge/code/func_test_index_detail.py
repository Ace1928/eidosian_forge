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
def test_index_detail(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'index', True)
    res_name = 'WikiDatabase'
    stack_identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '1')
    res_identity = identifier.ResourceIdentifier(resource_name=res_name, **stack_identity)
    req = self._get(stack_identity._tenant_path() + '/resources', {'with_detail': 'true'})
    resp_parameters = ({'OS::project_id': '3ab5b02fa01f4f95afa1e254afc4a435', 'network': 'cf05086d-07c7-4ed6-95e5-e4af724677e6', 'OS::stack_name': 's1', 'admin_pass': '******', 'key_name': 'kk', 'image': 'fa5d387e-541f-4dfb-ae8a-83a614683f84', 'db_port': '50000', 'OS::stack_id': '723d7cee-46b3-4433-9c21-f3378eb0bfc4', 'flavor': '1'},)
    engine_resp = [{u'resource_identity': dict(res_identity), u'stack_name': stack_identity.stack_name, u'resource_name': res_name, u'resource_status_reason': None, u'updated_time': u'2012-07-23T13:06:00Z', u'stack_identity': stack_identity, u'resource_action': u'CREATE', u'resource_status': u'COMPLETE', u'physical_resource_id': u'a3455d8c-9f88-404d-a85b-5315293e67de', u'resource_type': u'AWS::EC2::Instance', u'parameters': resp_parameters, u'description': u'Hello description', u'stack_user_project_id': u'6f38bcfebbc4400b82d50c1a2ea3057d'}]
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=engine_resp)
    result = self.controller.index(req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id)
    expected = {'resources': [{'links': [{'href': self._url(res_identity), 'rel': 'self'}, {'href': self._url(stack_identity), 'rel': 'stack'}], u'resource_name': res_name, u'logical_resource_id': res_name, u'resource_status_reason': None, u'updated_time': u'2012-07-23T13:06:00Z', u'resource_status': u'CREATE_COMPLETE', u'physical_resource_id': u'a3455d8c-9f88-404d-a85b-5315293e67de', u'resource_type': u'AWS::EC2::Instance', u'parameters': resp_parameters, u'description': u'Hello description', u'stack_user_project_id': u'6f38bcfebbc4400b82d50c1a2ea3057d'}]}
    self.assertEqual(expected, result)
    mock_call.assert_called_once_with(req.context, ('list_stack_resources', {'stack_identity': stack_identity, 'nested_depth': 0, 'with_detail': True, 'filters': {}}), version='1.25')