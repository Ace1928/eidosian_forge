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
def lookup(self, keys, name=None):
    """Looks up `keys` in a table, outputs the corresponding values."""
    if keys.dtype.base_dtype != self._key_dtype:
        raise TypeError('Signature mismatch. Keys must be dtype %s, got %s.' % (self._key_dtype, keys.dtype))
    self._check_keys(keys)
    num_shards = self._num_shards
    if num_shards == 1:
        return self._table_shards[0].lookup(keys, name=name)
    shard_indices = self._shard_indices(keys)
    key_shards = tf.dynamic_partition(keys, shard_indices, num_shards)
    value_shards = [self._table_shards[i].lookup(key_shards[i], name=name) for i in range(num_shards)]
    num_keys = tf.compat.v1.shape(keys)[0]
    original_indices = tf.range(num_keys)
    partitioned_indices = tf.dynamic_partition(original_indices, shard_indices, num_shards)
    return tf.dynamic_stitch(partitioned_indices, value_shards)