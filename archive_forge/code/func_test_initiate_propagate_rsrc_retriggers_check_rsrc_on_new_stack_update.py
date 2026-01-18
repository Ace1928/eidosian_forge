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
@mock.patch.object(check_resource.CheckResource, 'retrigger_check_resource')
@mock.patch.object(stack.Stack, 'load')
def test_initiate_propagate_rsrc_retriggers_check_rsrc_on_new_stack_update(self, mock_stack_load, mock_rcr, mock_cru, mock_crc, mock_pcr, mock_csc):
    key = sync_point.make_key(self.resource.id, self.stack.current_traversal, self.is_update)
    mock_pcr.side_effect = exception.EntityNotFound(entity='Sync Point', name=key)
    updated_stack = stack.Stack(self.ctx, self.stack.name, self.stack.t, self.stack.id, current_traversal='some_newy_trvl_uuid')
    mock_stack_load.return_value = updated_stack
    self.cr._initiate_propagate_resource(self.ctx, self.resource.id, self.stack.current_traversal, self.is_update, self.resource, self.stack)
    mock_rcr.assert_called_once_with(self.ctx, self.resource.id, updated_stack)