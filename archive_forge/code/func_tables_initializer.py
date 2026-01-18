import collections
import functools
import uuid
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.gen_lookup_ops import *
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as trackable_base
from tensorflow.python.trackable import resource
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.util import compat as compat_util
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['initializers.tables_initializer', 'tables_initializer'])
def tables_initializer(name='init_all_tables'):
    """Returns an Op that initializes all tables of the default graph.

  Args:
    name: Optional name for the initialization op.

  Returns:
    An Op that initializes all tables.  Note that if there are
    not tables the returned Op is a NoOp.

  @compatibility(TF2)
  `tf.compat.v1.tables_initializer` is no longer needed with eager execution and
  `tf.function`. In TF2, when creating an initializable table like a
  `tf.lookup.StaticHashTable`, the table will automatically be initialized on
  creation.

  #### Before & After Usage Example

  Before:

  >>> with tf.compat.v1.Session():
  ...   init = tf.compat.v1.lookup.KeyValueTensorInitializer(['a', 'b'], [1, 2])
  ...   table = tf.compat.v1.lookup.StaticHashTable(init, default_value=-1)
  ...   tf.compat.v1.tables_initializer().run()
  ...   result = table.lookup(tf.constant(['a', 'c'])).eval()
  >>> result
  array([ 1, -1], dtype=int32)

  After:

  >>> init = tf.lookup.KeyValueTensorInitializer(['a', 'b'], [1, 2])
  >>> table = tf.lookup.StaticHashTable(init, default_value=-1)
  >>> table.lookup(tf.constant(['a', 'c'])).numpy()
  array([ 1, -1], dtype=int32)

  @end_compatibility
  """
    initializers = ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS)
    if initializers:
        return control_flow_ops.group(*initializers, name=name)
    return control_flow_ops.no_op(name=name)