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
def test_HierarchicalTimer(self):
    RES = 0.01
    timer = HierarchicalTimer()
    start_time = time.perf_counter()
    timer.start('all')
    time.sleep(0.02)
    for i in range(10):
        timer.start('a')
        time.sleep(0.01)
        for j in range(5):
            timer.start('aa')
            time.sleep(0.001)
            timer.stop('aa')
        timer.start('ab')
        timer.stop('ab')
        timer.stop('a')
    end_time = time.perf_counter()
    timer.stop('all')
    ref = 'Identifier        ncalls   cumtime   percall      %\n---------------------------------------------------\nall                    1     [0-9.]+ +[0-9.]+ +100.0\n     ----------------------------------------------\n     a                10     [0-9.]+ +[0-9.]+ +[0-9.]+\n          -----------------------------------------\n          aa          50     [0-9.]+ +[0-9.]+ +[0-9.]+\n          ab          10     [0-9.]+ +[0-9.]+ +[0-9.]+\n          other      n/a     [0-9.]+ +n/a +[0-9.]+\n          =========================================\n     other           n/a     [0-9.]+ +n/a +[0-9.]+\n     ==============================================\n===================================================\n'.splitlines()
    for l, r in zip(str(timer).splitlines(), ref):
        self.assertRegex(l, r)
    self.assertEqual(1, timer.get_num_calls('all'))
    self.assertAlmostEqual(end_time - start_time, timer.get_total_time('all'), delta=RES)
    self.assertEqual(100.0, timer.get_relative_percent_time('all'))
    self.assertTrue(100.0 > timer.get_relative_percent_time('all.a'))
    self.assertTrue(50.0 < timer.get_relative_percent_time('all.a'))