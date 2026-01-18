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
def insert_or_assign(self, keys, values, name=None):
    """Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the table's
        key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """
    with ops.name_scope(name, '%s_lookup_table_insert' % self.name, [self.resource_handle, keys, values]):
        keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name='keys')
        values = ops.convert_to_tensor(values, dtype=self._value_dtype, name='values')
        with ops.colocate_with(self.resource_handle):
            op = gen_lookup_ops.lookup_table_insert_v2(self.resource_handle, keys, values)
        return op