import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def block_detect_pre(self):
    tmp = signal.signal(signal.SIGALRM, alarm_handler)
    if tmp != alarm_handler:
        self._old_signal_handler = tmp
    arm_alarm(self.debug_blocking_resolution)