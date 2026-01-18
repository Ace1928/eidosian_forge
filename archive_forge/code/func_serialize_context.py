import eventlet
from oslo_config import cfg
import oslo_messaging
from oslo_messaging.rpc import dispatcher
from osprofiler import profiler
from heat.common import context
@staticmethod
def serialize_context(ctxt):
    _context = ctxt.to_dict()
    prof = profiler.get()
    if prof:
        trace_info = {'hmac_key': prof.hmac_key, 'base_id': prof.get_base_id(), 'parent_id': prof.get_id()}
        _context.update({'trace_info': trace_info})
    return _context