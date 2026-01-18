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
def test_handle_stack_timeout(self, mock_cru, mock_crc, mock_pcr, mock_csc):
    mock_mf = self.stack.mark_failed = mock.Mock(return_value=True)
    self.cr._handle_stack_timeout(self.ctx, self.stack)
    mock_mf.assert_called_once_with(u'Timed out')