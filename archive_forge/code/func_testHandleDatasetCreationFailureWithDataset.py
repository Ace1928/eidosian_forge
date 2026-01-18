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
def testHandleDatasetCreationFailureWithDataset(self):
    model = Model(self.cluster_coord)
    restart_thread = self._restart_in_thread(5, 'worker')
    model.schedule_training_functions(3)
    model.rebuild_iterators(use_dataset_fn=False)
    model.schedule_training_functions(3)
    model.rebuild_iterators(use_dataset_fn=False)
    model.schedule_training_functions(3)
    model.join_training_functions()
    self.thread_coord.join([restart_thread])
    self.assertGreaterEqual(model.iterations.numpy(), 3)