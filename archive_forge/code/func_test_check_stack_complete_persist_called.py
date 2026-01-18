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
@mock.patch.object(dependencies.Dependencies, 'roots')
@mock.patch.object(stack.Stack, '_persist_state')
def test_check_stack_complete_persist_called(self, mock_persist_state, mock_dep_roots):
    mock_dep_roots.return_value = [(1, True)]
    check_resource.check_stack_complete(self.ctx, self.stack, self.stack.current_traversal, 1, self.stack.convergence_dependencies, True)
    self.assertTrue(mock_persist_state.called)