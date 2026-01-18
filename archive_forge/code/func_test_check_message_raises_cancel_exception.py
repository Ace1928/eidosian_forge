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
def test_check_message_raises_cancel_exception(self):
    msg_queue = eventlet.queue.LightQueue()
    msg_queue.put_nowait(rpc_api.THREAD_CANCEL)
    self.assertRaises(check_resource.CancelOperation, check_resource._check_for_message, msg_queue)