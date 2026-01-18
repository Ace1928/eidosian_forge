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
@mock.patch.object(stack.Stack, 'has_timed_out')
@mock.patch.object(check_resource.CheckResource, '_handle_stack_timeout')
def test_check_resource_handles_timeout(self, mock_hst, mock_to, mock_cru, mock_crc, mock_pcr, mock_csc):
    mock_to.return_value = True
    self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, {})
    self.assertTrue(mock_hst.called)