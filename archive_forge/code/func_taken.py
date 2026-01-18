import time as _time
from typing import Callable
from heapq import heappop as _heappop
from heapq import heappush as _heappush
from heapq import heappushpop as _heappushpop
from operator import attrgetter as _attrgetter
from collections import deque as _deque
def taken(ts, e):
    """Check if `ts` has already got an item scheduled nearby."""
    for item in self._schedule_interval_items:
        if abs(item.next_ts - ts) <= e:
            return True
        elif item.next_ts > ts + e:
            return False
    return False