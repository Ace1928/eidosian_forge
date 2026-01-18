import os
import uuid
from tensorflow.python.eager import test
from tensorflow.python.platform import flags
from tensorflow.python.profiler import profiler_v2 as profiler
def run_with_xprof(self, enable_python_trace, run_benchmark, func, num_iters_xprof, execution_mode, suid):
    if enable_python_trace:
        options = profiler.ProfilerOptions(python_tracer_level=1)
        logdir = os.path.join(flags.FLAGS.logdir, suid + '_with_python')
    else:
        options = profiler.ProfilerOptions(python_tracer_level=0)
        logdir = os.path.join(flags.FLAGS.logdir, suid)
    with profiler.Profile(logdir, options):
        total_time = run_benchmark(func, num_iters_xprof, execution_mode)
    us_per_example = float('{0:.3f}'.format(total_time * 1000000.0 / num_iters_xprof))
    return (logdir, us_per_example)