from time import monotonic
from kombu.asynchronous import timer as _timer
from . import base
def apply_timeout(target, args=(), kwargs=None, callback=None, accept_callback=None, pid=None, timeout=None, timeout_callback=None, Timeout=Timeout, apply_target=base.apply_target, **rest):
    kwargs = {} if not kwargs else kwargs
    try:
        with Timeout(timeout):
            return apply_target(target, args, kwargs, callback, accept_callback, pid, propagate=(Timeout,), **rest)
    except Timeout:
        return timeout_callback(False, timeout)