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
def test_check_resource_does_not_propagate_on_cancelling_cleanup(self, mock_cru, mock_crc, mock_pcr, mock_csc):
    mock_crc.side_effect = check_resource.CancelOperation
    self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, {})
    self.assertFalse(mock_pcr.called)
    self.assertFalse(mock_csc.called)