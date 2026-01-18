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
def dequeue_up_to(self, n, name=None):
    """Dequeues and concatenates `n` elements from this queue.

    **Note** This operation is not supported by all queues.  If a queue does not
    support DequeueUpTo, then a `tf.errors.UnimplementedError` is raised.

    This operation concatenates queue-element component tensors along
    the 0th dimension to make a single component tensor. If the queue
    has not been closed, all of the components in the dequeued tuple
    will have size `n` in the 0th dimension.

    If the queue is closed and there are more than `0` but fewer than
    `n` elements remaining, then instead of raising a
    `tf.errors.OutOfRangeError` like `tf.QueueBase.dequeue_many`,
    less than `n` elements are returned immediately.  If the queue is
    closed and there are `0` elements left in the queue, then a
    `tf.errors.OutOfRangeError` is raised just like in `dequeue_many`.
    Otherwise the behavior is identical to `dequeue_many`.

    Args:
      n: A scalar `Tensor` containing the number of elements to dequeue.
      name: A name for the operation (optional).

    Returns:
      The tuple of concatenated tensors that was dequeued.
    """
    if name is None:
        name = '%s_DequeueUpTo' % self._name
    ret = gen_data_flow_ops.queue_dequeue_up_to_v2(self._queue_ref, n=n, component_types=self._dtypes, name=name)
    if not context.executing_eagerly():
        op = ret[0].op
        for output, shape in zip(op.values(), self._shapes):
            output.set_shape(tensor_shape.TensorShape([None]).concatenate(shape))
    return self._dequeue_return_value(ret)