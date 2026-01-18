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
def test_index_enforces_global_index_if_global_tenant(self, mock_enforce):
    params = {'global_tenant': 'True'}
    req = self._get('/stacks', params=params)
    rpc_client = self.controller.rpc_client
    rpc_client.list_stacks = mock.Mock(return_value=[])
    rpc_client.count_stacks = mock.Mock()
    self.controller.index(req, tenant_id=self.tenant)
    mock_enforce.assert_called_with(action='global_index', scope=self.controller.REQUEST_SCOPE, is_registered_policy=True, context=self.context, target={'project_id': self.tenant})