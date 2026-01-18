import collections
import contextlib
import copy
import enum  # pylint: disable=g-bad-import-order
import functools
import threading
import weakref
import six
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context as eager_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def batch_reduce_to(self, reduce_op, value_destination_pairs, options=None):
    """Combine multiple `reduce_to` calls into one for faster execution.

    Similar to `reduce_to`, but accepts a list of (value, destinations) pairs.
    It's more efficient than reduce each value separately.

    This API currently can only be called in cross-replica context. Other
    variants to reduce values across replicas are:
    * `tf.distribute.StrategyExtended.reduce_to`: the non-batch version of
      this API.
    * `tf.distribute.ReplicaContext.all_reduce`: the counterpart of this API
      in replica context. It supports both batched and non-batched all-reduce.
    * `tf.distribute.Strategy.reduce`: a more convenient method to reduce
      to the host in cross-replica context.

    See `reduce_to` for more information.

    @tf.function
    def step_fn(var):

      def merge_fn(strategy, value, var):
        # All-reduce the value. Note that `value` here is a
        # `tf.distribute.DistributedValues`.
        reduced = strategy.extended.batch_reduce_to(
            tf.distribute.ReduceOp.SUM, [(value, var)])[0]
        strategy.extended.update(var, lambda var, value: var.assign(value),
            args=(reduced,))

      value = tf.identity(1.)
      tf.distribute.get_replica_context().merge_call(merge_fn,
        args=(value, var))

    def run(strategy):
      with strategy.scope():
        v = tf.Variable(0.)
        strategy.run(step_fn, args=(v,))
        return v

    run(tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"]))
    MirroredVariable:{
      0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>,
      1: <tf.Variable 'Variable/replica_1:0' shape=() dtype=float32, numpy=2.0>
    }
    run(tf.distribute.experimental.CentralStorageStrategy(
        compute_devices=["GPU:0", "GPU:1"], parameter_device="CPU:0"))
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>
    run(tf.distribute.OneDeviceStrategy("GPU:0"))
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>

    Args:
      reduce_op: a `tf.distribute.ReduceOp` value specifying how values should
        be combined. Allows using string representation of the enum such as
        "SUM", "MEAN".
      value_destination_pairs: a sequence of (value, destinations) pairs. See
        `tf.distribute.Strategy.reduce_to` for descriptions.
      options: a `tf.distribute.experimental.CommunicationOptions`. Options to
        perform collective operations. This overrides the default options if the
        `tf.distribute.Strategy` takes one in the constructor. See
        `tf.distribute.experimental.CommunicationOptions` for details of the
        options.

    Returns:
      A list of reduced values, one per pair in `value_destination_pairs`.
    """
    with monitoring.MonitoredTimer(distributed_api_time_counter.get_cell(self.__class__.__name__, 'Batch_reduce_to_eagerly')) if not ops.inside_function() else contextlib.nullcontext():
        if options is None:
            options = collective_util.Options()
        _require_cross_replica_or_default_context_extended(self)
        assert not isinstance(reduce_op, variable_scope.VariableAggregation)
        if isinstance(reduce_op, six.string_types):
            reduce_op = reduce_util.ReduceOp(reduce_op.upper())
        return self._batch_reduce_to(reduce_op, value_destination_pairs, options)