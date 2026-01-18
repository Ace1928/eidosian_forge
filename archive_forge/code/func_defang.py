import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def defang(self):
    self.cb = closed_callback
    if self.mark_as_closed is not None:
        self.mark_as_closed()
    self.spent = True