import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
def invoke_excepthook(thread):
    global excepthook
    try:
        hook = excepthook
        if hook is None:
            hook = old_excepthook
        args = ExceptHookArgs([*sys_exc_info(), thread])
        hook(args)
    except Exception as exc:
        exc.__suppress_context__ = True
        del exc
        if local_sys is not None and local_sys.stderr is not None:
            stderr = local_sys.stderr
        else:
            stderr = thread._stderr
        local_print('Exception in threading.excepthook:', file=stderr, flush=True)
        if local_sys is not None and local_sys.excepthook is not None:
            sys_excepthook = local_sys.excepthook
        else:
            sys_excepthook = old_sys_excepthook
        sys_excepthook(*sys_exc_info())
    finally:
        args = None