import sys
import heapq
import collections
import traceback
from eventlet.event import Event
from eventlet.greenthread import getcurrent
from eventlet.hubs import get_hub
import queue as Stdlib_Queue
from eventlet.timeout import Timeout
def getting(self):
    """Returns the number of greenthreads that are blocked waiting on an
        empty queue."""
    return len(self.getters)