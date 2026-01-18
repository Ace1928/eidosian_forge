import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def close_one(self):
    """ Triggered from the main run loop. If a listener's underlying FD was
            closed somehow, throw an exception back to the trampoline, which should
            be able to manage it appropriately.
        """
    listener = self.closed.pop()
    if not listener.greenlet.dead:
        listener.tb(eventlet.hubs.IOClosed(errno.ENOTCONN, 'Operation on closed file'))