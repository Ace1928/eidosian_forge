import webob.dec
from osprofiler import _utils as utils
from osprofiler import profiler
def get_trace_id_headers():
    """Adds the trace id headers (and any hmac) into provided dictionary."""
    p = profiler.get()
    if p and p.hmac_key:
        data = {'base_id': p.get_base_id(), 'parent_id': p.get_id()}
        pack = utils.signed_pack(data, p.hmac_key)
        return {X_TRACE_INFO: pack[0], X_TRACE_HMAC: pack[1]}
    return {}