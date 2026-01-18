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
@mock.patch.object(stack.Stack, 'time_remaining')
def test_is_cleanup_traversal_raise_update_inprogress(self, tr, mock_cru, mock_crc, mock_pcr, mock_csc):
    mock_crc.side_effect = exception.UpdateInProgress
    tr.return_value = 317
    self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
    mock_crc.assert_called_once_with(self.resource, self.resource.stack.t.id, self.worker.engine_id, tr(), mock.ANY)
    self.assertFalse(mock_cru.called)
    self.assertFalse(mock_pcr.called)
    self.assertFalse(mock_csc.called)