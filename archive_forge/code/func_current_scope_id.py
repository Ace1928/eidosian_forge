import contextlib
import threading
def current_scope_id():
    global _current_scope_id
    if not hasattr(_current_scope_id, 'value'):
        _current_scope_id.value = 1
    return _current_scope_id.value