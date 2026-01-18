import time
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.eager import context
from tensorflow.python.platform import test
def run_op_benchmark(self, op, iters=1, warmup=True, session_config=None):
    """Benchmarks the op.

    Runs the op `iters` times. In each iteration, the benchmark measures
    the time it takes to go execute the op.

    Args:
      op: The tf op to benchmark.
      iters: Number of times to repeat the timing.
      warmup: If true, warms up the session caches by running an untimed run.
      session_config: A ConfigProto protocol buffer with configuration options
        for the session. Applicable only for benchmarking in graph mode.

    Returns:
      A float, representing the per-execution wall time of the op in seconds.
      This is the median time (with respect to `iters`) it takes for the op
      to be executed `iters` num of times.
    """
    if context.executing_eagerly():
        return self._run_eager_benchmark(iterable=op, iters=iters, warmup=warmup)
    return self._run_graph_benchmark(iterable=op, iters=iters, warmup=warmup, session_config=session_config)