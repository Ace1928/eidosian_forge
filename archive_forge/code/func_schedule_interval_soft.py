import time as _time
from typing import Callable
from heapq import heappop as _heappop
from heapq import heappush as _heappush
from heapq import heappushpop as _heappushpop
from operator import attrgetter as _attrgetter
from collections import deque as _deque
def schedule_interval_soft(self, func, interval, *args, **kwargs):
    """Schedule a function to be called every ``interval`` seconds.

        This method is similar to `schedule_interval`, except that the
        clock will move the interval out of phase with other scheduled
        functions in order to distribute CPU load more evenly.

        This is useful for functions that need to be called regularly,
        but not relative to the initial start time.  :py:mod:`pyglet.media`
        does this for scheduling audio buffer updates, which need to occur
        regularly -- if all audio updates are scheduled at the same time
        (for example, mixing several tracks of a music score, or playing
        multiple videos back simultaneously), the resulting load on the
        CPU is excessive for those intervals but idle outside.  Using
        the soft interval scheduling, the load is more evenly distributed.

        Soft interval scheduling can also be used as an easy way to schedule
        graphics animations out of phase; for example, multiple flags
        waving in the wind.

        .. versionadded:: 1.1

        :Parameters:
            `func` : callable
                The function to call when the timer lapses.
            `interval` : float
                The number of seconds to wait between each call.

        """
    next_ts = self._get_soft_next_ts(self._get_nearest_ts(), interval)
    last_ts = next_ts - interval
    item = _ScheduledIntervalItem(func, interval, last_ts, next_ts, args, kwargs)
    _heappush(self._schedule_interval_items, item)