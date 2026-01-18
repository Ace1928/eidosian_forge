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
@mock.patch.object(resource.Resource, 'delete_convergence')
def test_check_resource_cleanup_delete(self, mock_delete):
    self.resource.current_template_id = 'new-template-id'
    check_resource.check_resource_cleanup(self.resource, self.resource.stack.t.id, 'engine-id', self.stack.timeout_secs(), None)
    self.assertTrue(mock_delete.called)