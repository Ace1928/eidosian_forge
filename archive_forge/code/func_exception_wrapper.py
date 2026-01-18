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
@functools.wraps(func)
def exception_wrapper(*args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        with excutils.save_and_reraise_exception():
            LOG.exception('Unhandled exception in %s', func.__name__)