import pyomo.common.unittest as unittest
from pyomo.common.tee import capture_output
import gc
from io import StringIO
from itertools import zip_longest
import logging
import sys
import time
from pyomo.common.log import LoggingIntercept
from pyomo.common.timing import (
from pyomo.environ import (
from pyomo.core.base.var import _VarData
def make_singleton_timer(self):
    timer = HierarchicalTimer()
    timer.start('root')
    timer.stop('root')
    timer.timers['root'].total_time = 5.0
    return timer