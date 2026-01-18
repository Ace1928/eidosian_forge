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
def test_generate_template(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'generate_template', True)
    req = self._get('/resource_types/TEST_TYPE/template')
    engine_response = {'Type': 'TEST_TYPE'}
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=engine_response)
    self.controller.generate_template(req, tenant_id=self.tenant, type_name='TEST_TYPE')
    mock_call.assert_called_once_with(req.context, ('generate_template', {'type_name': 'TEST_TYPE', 'template_type': 'cfn'}), version='1.9')