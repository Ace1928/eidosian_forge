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
def test_index_with_admin_context(self, mock_enforce):
    rpc_client = self.controller.rpc_client
    rpc_client.list_stacks = mock.Mock(return_value=[])
    rpc_client.count_stacks = mock.Mock()
    view_collection_mock = self.patchobject(stacks_view, 'collection')
    req = self._get('/stacks')
    req.context.is_admin = True
    self.controller.index(req, tenant_id=self.tenant)
    rpc_client.list_stacks.assert_called_once_with(mock.ANY, filters=mock.ANY)
    view_collection_mock.assert_called_once_with(mock.ANY, stacks=mock.ANY, count=mock.ANY, include_project=True)