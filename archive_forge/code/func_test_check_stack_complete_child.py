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
@mock.patch.object(sync_point, 'sync')
def test_check_stack_complete_child(self, mock_sync):
    check_resource.check_stack_complete(self.ctx, self.stack, self.stack.current_traversal, self.resource.id, self.stack.convergence_dependencies, True)
    self.assertFalse(mock_sync.called)