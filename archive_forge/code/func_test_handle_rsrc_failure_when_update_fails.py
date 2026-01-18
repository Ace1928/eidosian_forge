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
@mock.patch.object(stack.Stack, 'state_set')
@mock.patch.object(check_resource.CheckResource, 'retrigger_check_resource')
@mock.patch.object(stack.Stack, 'rollback')
def test_handle_rsrc_failure_when_update_fails(self, mock_tr, mock_rcr, mock_ss, mock_pdb, mock_cru, mock_crc, mock_pcr, mock_csc):
    mock_ss.return_value = False
    self.cr._handle_resource_failure(self.ctx, self.is_update, self.resource.id, self.stack, 'dummy-reason')
    self.assertTrue(mock_ss.called)
    self.assertFalse(mock_rcr.called)
    self.assertFalse(mock_pdb.called)
    self.assertFalse(mock_tr.called)