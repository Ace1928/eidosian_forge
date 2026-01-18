from collections import deque
import sys
from greenlet import GreenletExit
from eventlet import event
from eventlet import hubs
from eventlet import support
from eventlet import timeout
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
import warnings
def exc_after(seconds, *throw_args):
    warnings.warn('Instead of exc_after, which is deprecated, use Timeout(seconds, exception)', DeprecationWarning, stacklevel=2)
    if seconds is None:
        return timer.Timer(seconds, lambda: None)
    hub = hubs.get_hub()
    return hub.schedule_call_local(seconds, getcurrent().throw, *throw_args)