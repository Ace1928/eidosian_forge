import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def get_timers_count(hub):
    return len(hub.timers) + len(hub.next_timers)