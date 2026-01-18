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
def test_clear_except_base_timer(self):
    timer = HierarchicalTimer()
    timer.start('a')
    timer.start('b')
    timer.stop('b')
    timer.stop('a')
    timer.start('c')
    timer.stop('c')
    timer.start('d')
    timer.stop('d')
    timer.clear_except('b', 'c')
    key_set = set(timer.timers.keys())
    self.assertEqual(key_set, {'c'})