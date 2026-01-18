import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
def get_num_calls(self, identifier):
    """
        Parameters
        ----------
        identifier: str
            The full name of the timer including parent timers separated
            with dots.

        Returns
        -------
        n_calls: int
            The number of times start was called for the specified timer.
        """
    stack = identifier.split('.')
    timer = self._get_timer_from_stack(stack)
    return timer.n_calls