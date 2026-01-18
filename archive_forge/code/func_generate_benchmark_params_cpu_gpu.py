import timeit
import numpy as np
from keras.src import callbacks
from keras.src.benchmarks import distribution_util
def generate_benchmark_params_cpu_gpu(*params_list):
    """Extend the benchmark names with CPU and GPU suffix.

    Args:
      *params_list: A list of tuples represents the benchmark parameters.

    Returns:
      A list of strings with the benchmark name extended with CPU and GPU
      suffix.
    """
    benchmark_params = []
    for params in params_list:
        benchmark_params.extend([(param[0] + '_CPU',) + param[1:] for param in params])
        benchmark_params.extend([(param[0] + '_GPU',) + param[1:] for param in params])
    return benchmark_params