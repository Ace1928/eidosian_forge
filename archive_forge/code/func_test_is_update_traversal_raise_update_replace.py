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
@mock.patch.object(resource.Resource, 'load')
@mock.patch.object(resource.Resource, 'make_replacement')
@mock.patch.object(stack.Stack, 'time_remaining')
def test_is_update_traversal_raise_update_replace(self, tr, mock_mr, mock_load, mock_cru, mock_crc, mock_pcr, mock_csc):
    mock_load.return_value = (self.resource, self.stack, self.stack)
    mock_cru.side_effect = resource.UpdateReplace
    tr.return_value = 317
    self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
    mock_cru.assert_called_once_with(self.resource, self.resource.stack.t.id, set(), self.worker.engine_id, mock.ANY, mock.ANY)
    self.assertTrue(mock_mr.called)
    self.assertFalse(mock_crc.called)
    self.assertFalse(mock_pcr.called)
    self.assertFalse(mock_csc.called)