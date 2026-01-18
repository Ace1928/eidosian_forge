import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def squelch_timer_exception(self, timer, exc_info):
    if self.debug_exceptions:
        traceback.print_exception(*exc_info)
        sys.stderr.flush()