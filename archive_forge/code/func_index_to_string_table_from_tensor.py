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
def index_to_string_table_from_tensor(vocabulary_list, default_value='UNK', name=None):
    """Returns a lookup table that maps a `Tensor` of indices into strings.

  This operation constructs a lookup table to map int64 indices into string
  values. The mapping is initialized from a string `vocabulary_list` 1-D
  `Tensor` where each element is a value and the corresponding index within the
  tensor is the key.

  Any input which does not have a corresponding index in 'vocabulary_list'
  (an out-of-vocabulary entry) is assigned the `default_value`

  The underlying table must be initialized by calling
  `session.run(tf.compat.v1.tables_initializer())` or
  `session.run(table.init())` once.

  Elements in `vocabulary_list` cannot have duplicates, otherwise when executing
  the table initializer op, it will throw a `FailedPreconditionError`.

  Sample Usages:

  ```python
  vocabulary_list = tf.constant(["emerson", "lake", "palmer"])
  indices = tf.constant([1, 5], tf.int64)
  table = tf.lookup.index_to_string_table_from_tensor(
      vocabulary_list, default_value="UNKNOWN")
  values = table.lookup(indices)
  ...
  tf.compat.v1.tables_initializer().run()

  values.eval() ==> ["lake", "UNKNOWN"]
  ```

  Args:
    vocabulary_list: A 1-D string `Tensor` that specifies the strings to map
      from indices.
    default_value: The value to use for out-of-vocabulary indices.
    name: A name for this op (optional).

  Returns:
    The lookup table to map a string values associated to a given index `int64`
    `Tensors`.

  Raises:
    ValueError: when `vocabulary_list` is not set.
  """
    if vocabulary_list is None:
        raise ValueError('`vocabulary_list` argument must be specified.')
    with ops.name_scope(name, 'index_to_string'):
        vocabulary_list = ops.convert_to_tensor(vocabulary_list, dtypes.string)
        num_elements = array_ops.size(vocabulary_list)
        keys = math_ops.cast(math_ops.range(num_elements), dtypes.int64)
        init = KeyValueTensorInitializer(keys, vocabulary_list, dtypes.int64, dtypes.string, name='table_init')
        return StaticHashTableV1(init, default_value)