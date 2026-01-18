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
def index_table_from_file(vocabulary_file=None, num_oov_buckets=0, vocab_size=None, default_value=-1, hasher_spec=FastHashSpec, key_dtype=dtypes.string, name=None, key_column_index=TextFileIndex.WHOLE_LINE, value_column_index=TextFileIndex.LINE_NUMBER, delimiter='\t'):
    """Returns a lookup table that converts a string tensor into int64 IDs.

  This operation constructs a lookup table to convert tensor of strings into
  int64 IDs. The mapping can be initialized from a vocabulary file specified in
  `vocabulary_file`, where the whole line is the key and the zero-based line
  number is the ID.

  Any lookup of an out-of-vocabulary token will return a bucket ID based on its
  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the
  `default_value`.
  The bucket ID range is
  `[vocabulary size, vocabulary size + num_oov_buckets - 1]`.

  The underlying table must be initialized by calling
  `session.run(tf.compat.v1.tables_initializer())` or
  `session.run(table.init())` once.

  To specify multi-column vocabulary files, use key_column_index and
  value_column_index and delimiter.

  - TextFileIndex.LINE_NUMBER means use the line number starting from zero,
    expects data type int64.
  - TextFileIndex.WHOLE_LINE means use the whole line content, expects data
    type string.
  - A value >=0 means use the index (starting at zero) of the split line based
    on `delimiter`.

  Sample Usages:

  If we have a vocabulary file "test.txt" with the following content:

  ```
  emerson
  lake
  palmer
  ```

  ```python
  features = tf.constant(["emerson", "lake", "and", "palmer"])
  table = tf.lookup.index_table_from_file(
      vocabulary_file="test.txt", num_oov_buckets=1)
  ids = table.lookup(features)
  ...
  tf.compat.v1.tables_initializer().run()

  ids.eval()  ==> [0, 1, 3, 2]  # where 3 is the out-of-vocabulary bucket
  ```

  Args:
    vocabulary_file: The vocabulary filename, may be a constant scalar `Tensor`.
    num_oov_buckets: The number of out-of-vocabulary buckets.
    vocab_size: Number of the elements in the vocabulary, if known.
    default_value: The value to use for out-of-vocabulary feature values.
      Defaults to -1.
    hasher_spec: A `HasherSpec` to specify the hash function to use for
      assignation of out-of-vocabulary buckets.
    key_dtype: The `key` data type.
    name: A name for this op (optional).
    key_column_index: The column index from the text file to get the `key`
      values from. The default is to use the whole line content.
    value_column_index: The column index from the text file to get the `value`
      values from. The default is to use the line number, starting from zero.
    delimiter: The delimiter to separate fields in a line.

  Returns:
    The lookup table to map a `key_dtype` `Tensor` to index `int64` `Tensor`.

  Raises:
    ValueError: If `vocabulary_file` is not set.
    ValueError: If `num_oov_buckets` is negative or `vocab_size` is not greater
      than zero.
  """
    if vocabulary_file is None or (isinstance(vocabulary_file, str) and (not vocabulary_file)):
        raise ValueError('`vocabulary_file` must be specified and must not be empty.')
    if num_oov_buckets < 0:
        raise ValueError('num_oov_buckets must be greater or equal than 0, got %d.' % num_oov_buckets)
    if vocab_size is not None and vocab_size < 1:
        vocab_file_value = vocabulary_file
        if isinstance(vocabulary_file, tensor_lib.Tensor):
            vocab_file_value = tensor_util.constant_value(vocabulary_file) or '?'
        raise ValueError('`vocab_size` must be greater than 0, got %d for vocabulary_file: %s.' % (vocab_size, vocab_file_value))
    if not key_dtype.is_integer and dtypes.string != key_dtype.base_dtype:
        raise TypeError('Dtype for `keys` should be either integer or string.')
    with ops.name_scope(name, 'string_to_index'):
        table = None
        with ops.name_scope(None, 'hash_table'):
            init = TextFileIdTableInitializer(vocabulary_file, vocab_size=vocab_size, key_dtype=dtypes.int64 if key_dtype.is_integer else key_dtype, name='table_init', key_column_index=key_column_index, value_column_index=value_column_index, delimiter=delimiter)
            table = StaticHashTableV1(init, default_value)
        if num_oov_buckets:
            table = IdTableWithHashBuckets(table, num_oov_buckets=num_oov_buckets, hasher_spec=hasher_spec, key_dtype=key_dtype)
        return table