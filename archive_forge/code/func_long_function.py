import gc
import os
import sys
import threading
import time
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator as thread_coordinator
from tensorflow.python.training import server_lib
@def_function.function
def long_function():
    x = random_ops.random_uniform((1000, 1000))
    for _ in math_ops.range(10000):
        a = random_ops.random_uniform((1000, 1000))
        b = random_ops.random_uniform((1000, 1000))
        x += math_ops.matmul(a, b)
    return x