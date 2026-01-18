import json
from unittest import mock
from oslo_config import cfg
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.stacks as stacks
from heat.api.openstack.v1.views import stacks_view
from heat.common import context
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.common import urlfetch
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_adopt_timeout_not_int(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'create', True)
    identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '1')
    body = {'template': None, 'stack_name': identity.stack_name, 'parameters': {}, 'timeout_mins': 'not-an-int', 'adopt_stack_data': 'does not matter'}
    req = self._post('/stacks', json.dumps(body))
    mock_call = self.patchobject(rpc_client.EngineClient, 'call')
    ex = self.assertRaises(webob.exc.HTTPBadRequest, self.controller.create, req, tenant_id=self.tenant, body=body)
    self.assertEqual("Only integer is acceptable by 'timeout_mins'.", str(ex))
    mock_call.assert_not_called()