import time as _time
from typing import Callable
from heapq import heappop as _heappop
from heapq import heappush as _heappush
from heapq import heappushpop as _heappushpop
from operator import attrgetter as _attrgetter
from collections import deque as _deque
def schedule_interval_for_duration(self, func, interval, duration, *args, **kwargs):
    """Schedule a function to be called every `interval` seconds
        (see `schedule_interval`) and unschedule it after `duration` seconds.

        The callback function prototype is the same as for `schedule`.

        :Parameters:
            `func` : callable
                The function to call when the timer lapses.
            `interval` : float
                The number of seconds to wait between each call.
            `duration` : float
                The number of seconds for which the function is scheduled.

        """

    def _unschedule(dt: float, _func: Callable) -> None:
        self.unschedule(_func)
    self.schedule_interval(func, interval, *args, **kwargs)
    self.schedule_once(_unschedule, duration, func)