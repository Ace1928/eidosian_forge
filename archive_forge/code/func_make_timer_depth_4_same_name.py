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
def make_timer_depth_4_same_name(self):
    timer = HierarchicalTimer()
    timer.start('root')
    timer.start('a')
    timer.start('a')
    timer.start('a')
    timer.start('a')
    timer.stop('a')
    timer.stop('a')
    timer.stop('a')
    timer.stop('a')
    timer.stop('root')
    timer.timers['root'].total_time = 5.0
    timer.timers['root'].timers['a'].total_time = 1.0
    timer.timers['root'].timers['a'].timers['a'].total_time = 0.1
    timer.timers['root'].timers['a'].timers['a'].timers['a'].total_time = 0.01
    timer.timers['root'].timers['a'].timers['a'].timers['a'].timers['a'].total_time = 0.001
    return timer