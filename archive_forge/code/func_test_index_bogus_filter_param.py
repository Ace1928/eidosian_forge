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
def test_index_bogus_filter_param(self, mock_call, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'index', True)
    params = {'id': 'fake id', 'status': 'fake status', 'name': 'fake name', 'action': 'fake action', 'username': 'fake username', 'tenant': 'fake tenant', 'owner_id': 'fake owner-id', 'stack_name': 'fake stack name', 'stack_identity': 'fake identity', 'creation_time': 'create timestamp', 'updated_time': 'update timestamp', 'deletion_time': 'deletion timestamp', 'notification_topics': 'fake topic', 'description': 'fake description', 'template_description': 'fake description', 'parameters': 'fake params', 'outputs': 'fake outputs', 'stack_action': 'fake action', 'stack_status': 'fake status', 'stack_status_reason': 'fake status reason', 'capabilities': 'fake capabilities', 'disable_rollback': 'fake value', 'timeout_mins': 'fake timeout', 'stack_owner': 'fake owner', 'parent': 'fake parent', 'stack_user_project_id': 'fake project id', 'tags': 'fake tags', 'balrog': 'you shall not pass!'}
    req = self._get('/stacks', params=params)
    mock_call.return_value = []
    self.controller.index(req, tenant_id=self.tenant)
    rpc_call_args, _ = mock_call.call_args
    engine_args = rpc_call_args[1][1]
    self.assertIn('filters', engine_args)
    filters = engine_args['filters']
    self.assertEqual(16, len(filters))
    for key in ('id', 'status', 'name', 'action', 'username', 'tenant', 'owner_id', 'stack_name', 'stack_action', 'stack_status', 'stack_status_reason', 'disable_rollback', 'timeout_mins', 'stack_owner', 'parent', 'stack_user_project_id'):
        self.assertIn(key, filters)
    for key in ('stack_identity', 'creation_time', 'updated_time', 'deletion_time', 'notification_topics', 'description', 'template_description', 'parameters', 'outputs', 'capabilities', 'tags', 'balrog'):
        self.assertNotIn(key, filters)