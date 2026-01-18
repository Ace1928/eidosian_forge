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
def test_TicTocTimer_context_manager(self):
    SLEEP = 0.1
    RES = 0.05
    abs_time = time.perf_counter()
    with TicTocTimer() as timer:
        time.sleep(SLEEP)
    exclude = -time.perf_counter()
    time.sleep(SLEEP)
    exclude += time.perf_counter()
    with timer:
        time.sleep(SLEEP)
    abs_time = time.perf_counter() - abs_time
    self.assertGreater(abs_time, SLEEP * 3 - RES / 10)
    self.assertAlmostEqual(timer.toc(None), abs_time - exclude, delta=RES)