import eventlet.queue
import functools
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common import exception
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import sync_point
from heat.objects import resource as resource_objects
from heat.rpc import api as rpc_api
from heat.rpc import listener_client
def check_resource_update(rsrc, template_id, requires, engine_id, stack, msg_queue):
    """Create or update the Resource if appropriate."""
    check_message = functools.partial(_check_for_message, msg_queue)
    if rsrc.action == resource.Resource.INIT:
        rsrc.create_convergence(template_id, requires, engine_id, stack.time_remaining(), check_message)
    else:
        rsrc.update_convergence(template_id, requires, engine_id, stack.time_remaining(), stack, check_message)