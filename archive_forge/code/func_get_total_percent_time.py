import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
def get_total_percent_time(self, identifier):
    """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        percent_time: float
            The percent of time spent in the specified timer
            relative to the total time in all timers.
        """
    stack = identifier.split('.')
    timer = self._get_timer_from_stack(stack)
    total_time = 0
    for _timer in self.timers.values():
        total_time += _timer.total_time
    if total_time > 0:
        return timer.total_time / total_time * 100
    else:
        return float('nan')