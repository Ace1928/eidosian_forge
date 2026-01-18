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
@mock.patch.object(check_resource.CheckResource, '_handle_stack_timeout')
def test_do_check_resource_ignores_timeout_for_new_update(self, mock_hst, mock_cru, mock_crc, mock_pcr, mock_csc):
    mock_cru.side_effect = scheduler.Timeout(None, 60)
    self.is_update = True
    old_traversal = self.stack.current_traversal
    self.stack.current_traversal = 'new_traversal'
    self.cr._do_check_resource(self.ctx, old_traversal, self.stack.t, {}, self.is_update, self.resource, self.stack, {})
    self.assertFalse(mock_hst.called)