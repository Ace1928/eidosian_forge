import timeit
import numpy as np
from keras.src import callbacks
from keras.src.benchmarks import distribution_util
def get_benchmark_name(name):
    """Split the suffix of the benchmark name.

    For example, for the name = 'benchmark_layer_call__Conv2D_small_shape',
    the return value is ['Conv2D', 'small', 'shape'].

    This is to generate the metadata of the benchmark test.

    Args:
      name: A string, the benchmark name.

    Returns:
      A list of strings of the suffix in the benchmark name.
    """
    if '__' not in name or '_' not in name:
        raise ValueError('The format of the benchmark name is wrong.')
    return name.split('__')[-1].split('_')