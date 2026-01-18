import collections
import dataclasses
import functools
import io
import itertools
import threading
from absl import app
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.util import nest
def assert_sequential_execution(order, operations):
    """Asserts there's a deterministic execution order between the operations.

  Args:
    order: a map from a tf.Operation to its topological order.
    operations: a list of operations that should be executed sequentially. It
      can be given in any order.
  """
    operations = sorted(operations, key=lambda op: order[op])
    for i in range(len(operations) - 1):
        if not _exists_dependency(operations[i], operations[i + 1]):
            print(operations[i].graph.as_graph_def())
            raise AssertionError('No dependency between {} and {}. Graph is dumped to stdout.'.format(operations[i].name, operations[i + 1].name))