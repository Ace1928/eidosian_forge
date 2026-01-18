import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
def clear_except(self, *args):
    """Prune all "sub-timers" except those specified

        Parameters
        ----------
        args: str
            Keys that will be retained

        """
    if self.stack:
        raise RuntimeError('Cannot clear a HierarchicalTimer while any timers are active. Current active timer is %s. clear_except should only be called as a post-processing step.' % self.stack[-1])
    to_retain = set(args)
    _clear_timers_except(self, to_retain)