import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def dequeue(self, name=None):
    """Dequeues one element from this queue.

    If the queue is empty when this operation executes, it will block
    until there is an element to dequeue.

    At runtime, this operation may raise an error if the queue is
    `tf.QueueBase.close` before or during its execution. If the
    queue is closed, the queue is empty, and there are no pending
    enqueue operations that can fulfill this request,
    `tf.errors.OutOfRangeError` will be raised. If the session is
    `tf.Session.close`,
    `tf.errors.CancelledError` will be raised.

    Args:
      name: A name for the operation (optional).

    Returns:
      The tuple of tensors that was dequeued.
    """
    if name is None:
        name = '%s_Dequeue' % self._name
    if self._queue_ref.dtype == _dtypes.resource:
        ret = gen_data_flow_ops.queue_dequeue_v2(self._queue_ref, self._dtypes, name=name)
    else:
        ret = gen_data_flow_ops.queue_dequeue(self._queue_ref, self._dtypes, name=name)
    if not context.executing_eagerly():
        op = ret[0].op
        for output, shape in zip(op.values(), self._shapes):
            output.set_shape(shape)
    return self._dequeue_return_value(ret)