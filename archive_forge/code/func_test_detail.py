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
def test_detail(self, mock_call, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'detail', True)
    req = self._get('/stacks/detail')
    identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '1')
    engine_resp = [{u'stack_identity': dict(identity), u'updated_time': u'2012-07-09T09:13:11Z', u'template_description': u'blah', u'description': u'blah', u'stack_status_reason': u'Stack successfully created', u'creation_time': u'2012-07-09T09:12:45Z', u'stack_name': identity.stack_name, u'stack_action': u'CREATE', u'stack_status': u'COMPLETE', u'parameters': {'foo': 'bar'}, u'outputs': ['key', 'value'], u'notification_topics': [], u'capabilities': [], u'disable_rollback': True, u'timeout_mins': 60}]
    mock_call.return_value = engine_resp
    result = self.controller.detail(req, tenant_id=identity.tenant)
    expected = {'stacks': [{'links': [{'href': self._url(identity), 'rel': 'self'}], 'id': '1', u'updated_time': u'2012-07-09T09:13:11Z', u'template_description': u'blah', u'description': u'blah', u'stack_status_reason': u'Stack successfully created', u'creation_time': u'2012-07-09T09:12:45Z', u'stack_name': identity.stack_name, u'stack_status': u'CREATE_COMPLETE', u'parameters': {'foo': 'bar'}, u'outputs': ['key', 'value'], u'notification_topics': [], u'capabilities': [], u'disable_rollback': True, u'timeout_mins': 60}]}
    self.assertEqual(expected, result)
    default_args = {'limit': None, 'sort_keys': None, 'marker': None, 'sort_dir': None, 'filters': None, 'show_deleted': False, 'show_nested': False, 'show_hidden': False, 'tags': None, 'tags_any': None, 'not_tags': None, 'not_tags_any': None}
    mock_call.assert_called_once_with(req.context, ('list_stacks', default_args), version='1.33')