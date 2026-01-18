import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def prepare_timers(self):
    heappush = heapq.heappush
    t = self.timers
    for item in self.next_timers:
        if item[1].called:
            self.timers_canceled -= 1
        else:
            heappush(t, item)
    del self.next_timers[:]