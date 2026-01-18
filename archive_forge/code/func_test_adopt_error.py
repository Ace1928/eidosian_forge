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
def test_adopt_error(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'create', True)
    identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '1')
    parameters = {'app_dbx': 'test'}
    adopt_data = ['Test']
    body = {'template': None, 'stack_name': identity.stack_name, 'parameters': parameters, 'timeout_mins': 30, 'adopt_stack_data': str(adopt_data)}
    req = self._post('/stacks', json.dumps(body))
    resp = tools.request_with_middleware(fault.FaultWrapper, self.controller.create, req, tenant_id=self.tenant, body=body)
    self.assertEqual(400, resp.status_code)
    self.assertEqual('400 Bad Request', resp.status)
    self.assertIn('Invalid adopt data', resp.text)