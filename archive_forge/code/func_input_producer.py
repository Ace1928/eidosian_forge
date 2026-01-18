from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.summary import summary
from tensorflow.python.training import queue_runner
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.input_producer'])
@deprecation.deprecated(None, 'Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensor_slices(input_tensor).shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If `shuffle=False`, omit the `.shuffle(...)`.')
def input_producer(input_tensor, element_shape=None, num_epochs=None, shuffle=True, seed=None, capacity=32, shared_name=None, summary_name=None, name=None, cancel_op=None):
    """Output the rows of `input_tensor` to a queue for an input pipeline.

  Note: if `num_epochs` is not `None`, this function creates local counter
  `epochs`. Use `local_variables_initializer()` to initialize local variables.

  Args:
    input_tensor: A tensor with the rows to produce. Must be at least
      one-dimensional. Must either have a fully-defined shape, or
      `element_shape` must be defined.
    element_shape: (Optional.) A `TensorShape` representing the shape of a
      row of `input_tensor`, if it cannot be inferred.
    num_epochs: (Optional.) An integer. If specified `input_producer` produces
      each row of `input_tensor` `num_epochs` times before generating an
      `OutOfRange` error. If not specified, `input_producer` can cycle through
      the rows of `input_tensor` an unlimited number of times.
    shuffle: (Optional.) A boolean. If true, the rows are randomly shuffled
      within each epoch.
    seed: (Optional.) An integer. The seed to use if `shuffle` is true.
    capacity: (Optional.) The capacity of the queue to be used for buffering
      the input.
    shared_name: (Optional.) If set, this queue will be shared under the given
      name across multiple sessions.
    summary_name: (Optional.) If set, a scalar summary for the current queue
      size will be generated, using this name as part of the tag.
    name: (Optional.) A name for queue.
    cancel_op: (Optional.) Cancel op for the queue

  Returns:
    A queue with the output rows.  A `QueueRunner` for the queue is
    added to the current `QUEUE_RUNNER` collection of the current
    graph.

  Raises:
    ValueError: If the shape of the input cannot be inferred from the arguments.
    RuntimeError: If called with eager execution enabled.

  @compatibility(eager)
  Input pipelines based on Queues are not supported when eager execution is
  enabled. Please use the `tf.data` API to ingest data under eager execution.
  @end_compatibility
  """
    if context.executing_eagerly():
        raise RuntimeError('Input pipelines based on Queues are not supported when eager execution is enabled. Please use tf.data to ingest data into your model instead.')
    with ops.name_scope(name, 'input_producer', [input_tensor]):
        input_tensor = ops.convert_to_tensor(input_tensor, name='input_tensor')
        element_shape = input_tensor.shape[1:].merge_with(element_shape)
        if not element_shape.is_fully_defined():
            raise ValueError('Either `input_tensor` must have a fully defined shape or `element_shape` must be specified')
        if shuffle:
            input_tensor = random_ops.random_shuffle(input_tensor, seed=seed)
        input_tensor = limit_epochs(input_tensor, num_epochs)
        q = data_flow_ops.FIFOQueue(capacity=capacity, dtypes=[input_tensor.dtype.base_dtype], shapes=[element_shape], shared_name=shared_name, name=name)
        enq = q.enqueue_many([input_tensor])
        queue_runner.add_queue_runner(queue_runner.QueueRunner(q, [enq], cancel_op=cancel_op))
        if summary_name is not None:
            summary.scalar(summary_name, math_ops.cast(q.size(), dtypes.float32) * (1.0 / capacity))
        return q