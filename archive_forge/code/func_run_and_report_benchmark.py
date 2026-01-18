import time
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.eager import context
from tensorflow.python.platform import test
def run_and_report_benchmark(self, dataset, num_elements, name, iters=5, extras=None, warmup=True, apply_default_optimizations=False, session_config=None):
    """Benchmarks the dataset and reports the stats.

    Runs the dataset `iters` times. In each iteration, the benchmark measures
    the time it takes to go through `num_elements` elements of the dataset.
    This is followed by logging/printing the benchmark stats.

    Args:
      dataset: Dataset to benchmark.
      num_elements: Number of dataset elements to iterate through each benchmark
        iteration.
      name: Name of the benchmark.
      iters: Number of times to repeat the timing.
      extras: A dict which maps string keys to additional benchmark info.
      warmup: If true, warms up the session caches by running an untimed run.
      apply_default_optimizations: Determines whether default optimizations
        should be applied.
      session_config: A ConfigProto protocol buffer with configuration options
        for the session. Applicable only for benchmarking in graph mode.

    Returns:
      A float, representing the per-element wall time of the dataset in seconds.
      This is the median time (with respect to `iters`) it takes for the dataset
      to go through `num_elements` elements, divided by `num_elements.`
    """
    wall_time = self.run_benchmark(dataset=dataset, num_elements=num_elements, iters=iters, warmup=warmup, apply_default_optimizations=apply_default_optimizations, session_config=session_config)
    if extras is None:
        extras = {}
    if context.executing_eagerly():
        name = '{}.eager'.format(name)
        extras['implementation'] = 'eager'
    else:
        name = '{}.graph'.format(name)
        extras['implementation'] = 'graph'
    extras['num_elements'] = num_elements
    self.report_benchmark(wall_time=wall_time, iters=iters, name=name, extras=extras)
    return wall_time