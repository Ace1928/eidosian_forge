import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def squelch_exception(self, fileno, exc_info):
    traceback.print_exception(*exc_info)
    sys.stderr.write('Removing descriptor: %r\n' % (fileno,))
    sys.stderr.flush()
    try:
        self.remove_descriptor(fileno)
    except Exception as e:
        sys.stderr.write('Exception while removing descriptor! %r\n' % (e,))
        sys.stderr.flush()