import time as _time
from typing import Callable
from heapq import heappop as _heappop
from heapq import heappush as _heappush
from heapq import heappushpop as _heappushpop
from operator import attrgetter as _attrgetter
from collections import deque as _deque
def call_scheduled_functions(self, dt: float) -> bool:
    """Call scheduled functions that elapsed on the last `update_time`.

        Returns True if any functions were called, otherwise False.

        .. versionadded:: 1.2

        :Parameters:
            dt : float
                The elapsed time since the last update to pass to each
                scheduled function.  This is *not* used to calculate which
                functions have elapsed.
        """
    now = self.last_ts
    result = False
    if self._schedule_items:
        result = True
        for item in list(self._schedule_items):
            item.func(dt, *item.args, **item.kwargs)
    interval_items = self._schedule_interval_items
    try:
        if interval_items[0].next_ts > now:
            return result
    except IndexError:
        return result
    self._current_interval_item = item = None
    get_soft_next_ts = self._get_soft_next_ts
    while interval_items:
        if item is None:
            item = _heappop(interval_items)
        else:
            item = _heappushpop(interval_items, item)
        self._current_interval_item = item
        if item.next_ts > now:
            break
        item.func(now - item.last_ts, *item.args, **item.kwargs)
        if item.interval:
            item.next_ts = item.last_ts + item.interval
            item.last_ts = now
            if item.next_ts <= now:
                if now - item.next_ts < 0.05:
                    item.next_ts = now + item.interval
                else:
                    item.next_ts = get_soft_next_ts(now, item.interval)
                    item.last_ts = item.next_ts - item.interval
        else:
            self._current_interval_item = item = None
    if item is not None:
        _heappush(interval_items, item)
    return True