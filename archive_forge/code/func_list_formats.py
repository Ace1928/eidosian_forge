from collections import defaultdict
from ..core import Store
def list_formats(format_type, backend=None):
    """
    Returns list of supported formats for a particular
    backend.
    """
    if backend is None:
        backend = Store.current_backend
        mode = Store.renderers[backend].mode if backend in Store.renderers else None
    else:
        split = backend.split(':')
        backend, mode = split if len(split) == 2 else (split[0], 'default')
    if backend in Store.renderers:
        return Store.renderers[backend].mode_formats[format_type]
    else:
        return []