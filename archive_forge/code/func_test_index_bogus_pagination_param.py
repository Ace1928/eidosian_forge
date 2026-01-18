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
@mock.patch.object(rpc_client.EngineClient, 'call')
def test_index_bogus_pagination_param(self, mock_call, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'index', True)
    params = {'limit': 10, 'sort_keys': 'fake sort keys', 'marker': 'fake marker', 'sort_dir': 'fake sort dir', 'balrog': 'you shall not pass!'}
    req = self._get('/stacks', params=params)
    mock_call.return_value = []
    self.controller.index(req, tenant_id=self.tenant)
    rpc_call_args, _ = mock_call.call_args
    engine_args = rpc_call_args[1][1]
    self.assertEqual(12, len(engine_args))
    self.assertIn('limit', engine_args)
    self.assertIn('sort_keys', engine_args)
    self.assertIn('marker', engine_args)
    self.assertIn('sort_dir', engine_args)
    self.assertIn('filters', engine_args)
    self.assertNotIn('balrog', engine_args)