import time as _time
from typing import Callable
from heapq import heappop as _heappop
from heapq import heappush as _heappush
from heapq import heappushpop as _heappushpop
from operator import attrgetter as _attrgetter
from collections import deque as _deque
def schedule_once(self, func, delay, *args, **kwargs):
    """Schedule a function to be called once after `delay` seconds.

        The callback function prototype is the same as for `schedule`.

        :Parameters:
            `func` : callable
                The function to call when the timer lapses.
            `delay` : float
                The number of seconds to wait before the timer lapses.
        """
    last_ts = self._get_nearest_ts()
    next_ts = last_ts + delay
    item = _ScheduledIntervalItem(func, 0, last_ts, next_ts, args, kwargs)
    _heappush(self._schedule_interval_items, item)