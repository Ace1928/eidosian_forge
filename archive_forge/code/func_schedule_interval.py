import time as _time
from typing import Callable
from heapq import heappop as _heappop
from heapq import heappush as _heappush
from heapq import heappushpop as _heappushpop
from operator import attrgetter as _attrgetter
from collections import deque as _deque
def schedule_interval(self, func, interval, *args, **kwargs):
    """Schedule a function to be called every `interval` seconds.

        Specifying an interval of 0 prevents the function from being
        called again (see `schedule` to call a function as often as possible).

        The callback function prototype is the same as for `schedule`.

        :Parameters:
            `func` : callable
                The function to call when the timer lapses.
            `interval` : float
                The number of seconds to wait between each call.

        """
    last_ts = self._get_nearest_ts()
    next_ts = last_ts + interval
    item = _ScheduledIntervalItem(func, interval, last_ts, next_ts, args, kwargs)
    _heappush(self._schedule_interval_items, item)