from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.engine import check_resource
from heat.engine import dependencies
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import sync_point
from heat.engine import worker
from heat.rpc import api as rpc_api
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(resource.Resource, 'create_convergence')
@mock.patch.object(resource.Resource, 'update_convergence')
def test_check_resource_update_init_action(self, mock_update, mock_create):
    self.resource.action = 'INIT'
    check_resource.check_resource_update(self.resource, self.resource.stack.t.id, set(), 'engine-id', self.stack, None)
    self.assertTrue(mock_create.called)
    self.assertFalse(mock_update.called)