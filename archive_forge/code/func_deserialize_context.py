import eventlet
from oslo_config import cfg
import oslo_messaging
from oslo_messaging.rpc import dispatcher
from osprofiler import profiler
from heat.common import context
@staticmethod
def deserialize_context(ctxt):
    trace_info = ctxt.pop('trace_info', None)
    if trace_info:
        profiler.init(**trace_info)
    return context.RequestContext.from_dict(ctxt)