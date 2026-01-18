from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from six.moves import range
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.checkpoint import saveable_compat
def export(self, name=None):
    """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    with ops.name_scope(name, '%s_lookup_table_export_values' % self.name, [self.resource_handle]) as name:
        with ops.colocate_with(self.resource_handle):
            exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(self.resource_handle, self._key_dtype, self._value_dtype, name=name)
    return (exported_keys, exported_values)