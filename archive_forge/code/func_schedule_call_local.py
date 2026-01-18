import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def schedule_call_local(self, seconds, cb, *args, **kw):
    """Schedule a callable to be called after 'seconds' seconds have
        elapsed. Cancel the timer if greenlet has exited.
            seconds: The number of seconds to wait.
            cb: The callable to call after the given time.
            *args: Arguments to pass to the callable when called.
            **kw: Keyword arguments to pass to the callable when called.
        """
    t = timer.LocalTimer(seconds, cb, *args, **kw)
    self.add_timer(t)
    return t