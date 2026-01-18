import base64
import inspect
import builtins
def obj_python_attrs(msg_):
    """iterate object attributes for stringify purposes
    """
    if hasattr(msg_, '_fields'):
        for k in msg_._fields:
            yield (k, getattr(msg_, k))
        return
    base = getattr(msg_, '_base_attributes', [])
    opt = getattr(msg_, '_opt_attributes', [])
    for k, v in inspect.getmembers(msg_):
        if k in opt:
            pass
        elif k.startswith('_'):
            continue
        elif callable(v):
            continue
        elif k in base:
            continue
        elif hasattr(msg_.__class__, k):
            continue
        yield (k, v)