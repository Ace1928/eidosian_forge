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
def test_TicTocTimer_tictoc(self):
    SLEEP = 0.1
    RES = 0.02
    if 'pypy_version_info' in dir(sys):
        RES *= 2.5
    abs_time = time.perf_counter()
    timer = TicTocTimer()
    time.sleep(SLEEP)
    with capture_output() as out:
        start_time = time.perf_counter()
        timer.tic(None)
    self.assertEqual(out.getvalue(), '')
    with capture_output() as out:
        start_time = time.perf_counter()
        timer.tic()
    self.assertRegex(out.getvalue(), '\\[    [.0-9]+\\] Resetting the tic/toc delta timer')
    time.sleep(SLEEP)
    with capture_output() as out:
        ref = time.perf_counter()
        delta = timer.toc()
    self.assertAlmostEqual(ref - start_time, delta, delta=RES)
    self.assertRegex(out.getvalue(), '\\[\\+   [.0-9]+\\] .* in test_TicTocTimer_tictoc')
    with capture_output() as out:
        self.assertAlmostEqual(time.perf_counter() - ref, timer.toc(None), delta=RES)
    self.assertEqual(out.getvalue(), '')
    with capture_output() as out:
        ref = time.perf_counter()
        total = timer.toc(delta=False)
    self.assertAlmostEqual(ref - start_time, total, delta=RES)
    self.assertRegex(out.getvalue(), '\\[    [.0-9]+\\] .* in test_TicTocTimer_tictoc')
    ref *= -1
    time.sleep(SLEEP)
    ref += time.perf_counter()
    timer.stop()
    cumul_stop1 = timer.toc(None)
    self.assertAlmostEqual(ref, cumul_stop1, delta=RES)
    with self.assertRaisesRegex(RuntimeError, 'Stopping a TicTocTimer that was already stopped'):
        timer.stop()
    time.sleep(SLEEP)
    cumul_stop2 = timer.toc(None)
    self.assertEqual(cumul_stop1, cumul_stop2)
    ref -= time.perf_counter()
    timer.start()
    time.sleep(SLEEP)
    with capture_output() as out:
        ref += time.perf_counter()
        timer.stop()
        delta = timer.toc()
    self.assertAlmostEqual(ref, delta, delta=RES)
    self.assertRegex(out.getvalue(), '\\[    [.0-9]+\\|   1\\] .* in test_TicTocTimer_tictoc')
    with capture_output() as out:
        total = timer.toc(delta=False)
    self.assertAlmostEqual(delta, total, delta=RES)
    self.assertRegex(out.getvalue(), '\\[    [.0-9]+\\|   1\\] .* in test_TicTocTimer_tictoc')