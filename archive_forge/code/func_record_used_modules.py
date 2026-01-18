import sys
from typing import Any, Callable, Iterable, List, Tuple
def record_used_modules(frame, event, arg):
    if event != 'call':
        return
    name = frame.f_code.co_name
    module = None
    if name in frame.f_globals:
        module = frame.f_globals[name].__module__
    elif name in frame.f_locals:
        module = frame.f_locals[name].__module__
    elif 'self' in frame.f_locals:
        method = getattr(frame.f_locals['self'], name, None)
        module = method.__module__ if method else None
    if module:
        modules_used.add(module)