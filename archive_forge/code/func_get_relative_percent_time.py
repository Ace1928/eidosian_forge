import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
def get_relative_percent_time(self, identifier):
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
            relative to the timer's immediate parent.
        """
    stack = identifier.split('.')
    timer = self._get_timer_from_stack(stack)
    parent = self._get_timer_from_stack(stack[:-1])
    if parent is self:
        return self.get_total_percent_time(identifier)
    elif parent.total_time > 0:
        return timer.total_time / parent.total_time * 100
    else:
        return float('nan')