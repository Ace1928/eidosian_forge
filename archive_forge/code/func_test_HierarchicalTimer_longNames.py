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
def test_HierarchicalTimer_longNames(self):
    RES = 0.01
    timer = HierarchicalTimer()
    start_time = time.perf_counter()
    timer.start('all' * 25)
    time.sleep(0.02)
    for i in range(10):
        timer.start('a' * 75)
        time.sleep(0.01)
        for j in range(5):
            timer.start('aa' * 20)
            time.sleep(0.001)
            timer.stop('aa' * 20)
        timer.start('ab' * 20)
        timer.stop('ab' * 20)
        timer.stop('a' * 75)
    end_time = time.perf_counter()
    timer.stop('all' * 25)
    ref = ('Identifier%s   ncalls   cumtime   percall      %%\n%s------------------------------------\n%s%s        1     [0-9.]+ +[0-9.]+ +100.0\n    %s------------------------------------\n    %s%s       10     [0-9.]+ +[0-9.]+ +[0-9.]+\n        %s------------------------------------\n        %s%s       50     [0-9.]+ +[0-9.]+ +[0-9.]+\n        %s%s       10     [0-9.]+ +[0-9.]+ +[0-9.]+\n        other%s      n/a     [0-9.]+ +n/a +[0-9.]+\n        %s====================================\n    other%s      n/a     [0-9.]+ +n/a +[0-9.]+\n    %s====================================\n%s====================================\n' % (' ' * 69, '-' * 79, 'all' * 25, ' ' * 4, '-' * 75, 'a' * 75, '', '-' * 71, 'aa' * 20, ' ' * 31, 'ab' * 20, ' ' * 31, ' ' * 66, '=' * 71, ' ' * 70, '=' * 75, '=' * 79)).splitlines()
    for l, r in zip(str(timer).splitlines(), ref):
        self.assertRegex(l, r)