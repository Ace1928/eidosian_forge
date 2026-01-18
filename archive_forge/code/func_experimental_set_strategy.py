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
@tf_export('distribute.experimental_set_strategy')
def experimental_set_strategy(strategy):
    """Set a `tf.distribute.Strategy` as current without `with strategy.scope()`.

  ```
  tf.distribute.experimental_set_strategy(strategy1)
  f()
  tf.distribute.experimental_set_strategy(strategy2)
  g()
  tf.distribute.experimental_set_strategy(None)
  h()
  ```

  is equivalent to:

  ```
  with strategy1.scope():
    f()
  with strategy2.scope():
    g()
  h()
  ```

  In general, you should use the `with strategy.scope():` API, but this
  alternative may be convenient in notebooks where you would have to put
  each cell in a `with strategy.scope():` block.

  Note: This should only be called outside of any TensorFlow scope to
  avoid improper nesting.

  Args:
    strategy: A `tf.distribute.Strategy` object or None.

  Raises:
    RuntimeError: If called inside a `with strategy.scope():`.
  """
    old_scope = ops.get_default_graph()._global_distribute_strategy_scope
    if old_scope is not None:
        old_scope.__exit__(None, None, None)
        ops.get_default_graph()._global_distribute_strategy_scope = None
    if has_strategy():
        raise RuntimeError('Must not be called inside a `tf.distribute.Strategy` scope.')
    if strategy is not None:
        new_scope = strategy.scope()
        new_scope.__enter__()
        ops.get_default_graph()._global_distribute_strategy_scope = new_scope