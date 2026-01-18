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
@mock.patch.object(stack.Stack, 'purge_db')
def test_handle_failure(self, mock_purgedb, mock_cru, mock_crc, mock_pcr, mock_csc):
    self.stack.mark_failed('dummy-reason')
    mock_purgedb.assert_called_once_with()
    self.assertEqual('dummy-reason', self.stack.status_reason)