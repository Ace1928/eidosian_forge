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
def rebuild_iterators(self, use_dataset_fn=True):
    if use_dataset_fn:

        def dataset_fn():
            data = random_ops.random_uniform((10, 10))
            dataset = dataset_ops.DatasetV2.from_tensors([data]).repeat()
            return dataset

        def distribute_dataset_fn():
            return self.cluster_coord.strategy.distribute_datasets_from_function(lambda _: dataset_fn())
        self.iterator = iter(self.cluster_coord.create_per_worker_dataset(distribute_dataset_fn))
        self.iterator2 = iter(self.cluster_coord.create_per_worker_dataset(distribute_dataset_fn))
    else:
        data = random_ops.random_uniform((10, 10))
        dataset = dataset_ops.DatasetV2.from_tensors([data]).repeat()
        self.iterator = iter(self.cluster_coord.create_per_worker_dataset(dataset))
        self.iterator2 = iter(self.cluster_coord.create_per_worker_dataset(dataset))