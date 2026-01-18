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
def propagate_check_resource(cnxt, rpc_client, next_res_id, current_traversal, predecessors, sender_key, sender_data, is_update, adopt_stack_data):
    """Trigger processing of node if all of its dependencies are satisfied."""

    def do_check(entity_id, data):
        rpc_client.check_resource(cnxt, entity_id, current_traversal, data, is_update, adopt_stack_data)
    sync_point.sync(cnxt, next_res_id, current_traversal, is_update, do_check, predecessors, {sender_key: sender_data})