import sys
import heapq
import collections
import traceback
from eventlet.event import Event
from eventlet.greenthread import getcurrent
from eventlet.hubs import get_hub
import queue as Stdlib_Queue
from eventlet.timeout import Timeout
def putting(self):
    """Returns the number of greenthreads that are blocked waiting to put
        items into the queue."""
    return len(self.putters)