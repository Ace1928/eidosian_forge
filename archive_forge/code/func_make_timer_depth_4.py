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
def make_timer_depth_4(self):
    timer = HierarchicalTimer()
    timer.start('root')
    timer.start('a')
    timer.start('b')
    timer.stop('b')
    timer.start('c')
    timer.start('d')
    timer.start('e')
    timer.stop('e')
    timer.stop('d')
    timer.stop('c')
    timer.stop('a')
    timer.start('b')
    timer.start('c')
    timer.start('e')
    timer.stop('e')
    timer.stop('c')
    timer.start('d')
    timer.stop('d')
    timer.stop('b')
    timer.stop('root')
    timer.timers['root'].total_time = 5.0
    timer.timers['root'].timers['a'].total_time = 4.0
    timer.timers['root'].timers['a'].timers['b'].total_time = 1.1
    timer.timers['root'].timers['a'].timers['c'].total_time = 2.2
    timer.timers['root'].timers['a'].timers['c'].timers['d'].total_time = 0.9
    timer.timers['root'].timers['a'].timers['c'].timers['d'].timers['e'].total_time = 0.6
    timer.timers['root'].timers['b'].total_time = 0.88
    timer.timers['root'].timers['b'].timers['c'].total_time = 0.07
    timer.timers['root'].timers['b'].timers['c'].timers['e'].total_time = 0.04
    timer.timers['root'].timers['b'].timers['d'].total_time = 0.05
    return timer