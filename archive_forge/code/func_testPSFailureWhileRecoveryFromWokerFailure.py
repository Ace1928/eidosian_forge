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
def testPSFailureWhileRecoveryFromWokerFailure(self):
    model = self._create_model_and_run_indefinitely()
    time.sleep(1)
    self.assertFalse(self.cluster_coord.done())

    def kill(task):
        self._cluster.kill_task(task, 0)
        self.sleep(1)
        self._cluster.start_task(task, 0)
    kill_thread_1 = threading.Thread(target=kill, args=('worker',))
    kill_thread_2 = threading.Thread(target=kill, args=('ps',))
    kill_thread_1.start()
    kill_thread_2.start()
    kill_thread_1.join()
    kill_thread_2.join()
    with self.assertRaises((errors.UnavailableError, errors.InvalidArgumentError)):
        model.join_training_functions()