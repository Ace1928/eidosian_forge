import eventlet.queue
import functools
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import excutils
from oslo_utils import uuidutils
from osprofiler import profiler
from heat.common import context
from heat.common import messaging as rpc_messaging
from heat.db import api as db_api
from heat.engine import check_resource
from heat.engine import node_data
from heat.engine import stack as parser
from heat.engine import sync_point
from heat.objects import stack as stack_objects
from heat.rpc import api as rpc_api
from heat.rpc import worker_client as rpc_client
@context.request_context
@log_exceptions
def cancel_check_resource(self, cnxt, stack_id):
    """Cancel check_resource for given stack.

        All the workers running for the given stack will be
        cancelled.
        """
    _cancel_check_resource(stack_id, self.engine_id, self.thread_group_mgr)