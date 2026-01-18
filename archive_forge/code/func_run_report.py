import os
import uuid
from tensorflow.python.eager import test
from tensorflow.python.platform import flags
from tensorflow.python.profiler import profiler_v2 as profiler
def run_report(self, run_benchmark, func, num_iters, execution_mode=None):
    """Run and report benchmark results."""
    total_time = run_benchmark(func, num_iters, execution_mode)
    mean_us = total_time * 1000000.0 / num_iters
    extras = {'examples_per_sec': float('{0:.3f}'.format(num_iters / total_time)), 'us_per_example': float('{0:.3f}'.format(total_time * 1000000.0 / num_iters))}
    if flags.FLAGS.xprof:
        suid = str(uuid.uuid4())
        num_iters_xprof = min(100, num_iters)
        xprof_link, us_per_example = self.run_with_xprof(True, run_benchmark, func, num_iters_xprof, execution_mode, suid)
        extras['xprof link with python trace'] = xprof_link
        extras['us_per_example with xprof and python'] = us_per_example
        xprof_link, us_per_example = self.run_with_xprof(False, run_benchmark, func, num_iters_xprof, execution_mode, suid)
        extras['xprof link'] = xprof_link
        extras['us_per_example with xprof'] = us_per_example
    benchmark_name = self._get_benchmark_name()
    self.report_benchmark(iters=num_iters, wall_time=mean_us, extras=extras, name=benchmark_name)