import functools
import warnings
import threading
import sys
def probe_func(frame, event, arg):
    if event == 'return':
        locals = frame.f_locals
        func_locals.update(dict(((k, locals.get(k)) for k in keys)))
        sys.settrace(None)
    return probe_func