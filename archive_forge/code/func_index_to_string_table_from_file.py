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
def index_to_string_table_from_file(vocabulary_file, vocab_size=None, default_value='UNK', name=None, key_column_index=TextFileIndex.LINE_NUMBER, value_column_index=TextFileIndex.WHOLE_LINE, delimiter='\t'):
    """Returns a lookup table that maps a `Tensor` of indices into strings.

  This operation constructs a lookup table to map int64 indices into string
  values. The table is initialized from a vocabulary file specified in
  `vocabulary_file`, where the whole line is the value and the
  zero-based line number is the index.

  Any input which does not have a corresponding index in the vocabulary file
  (an out-of-vocabulary entry) is assigned the `default_value`

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
  indices = tf.constant([1, 5], tf.int64)
  table = tf.lookup.index_to_string_table_from_file(
      vocabulary_file="test.txt", default_value="UNKNOWN")
  values = table.lookup(indices)
  ...
  tf.compat.v1.tables_initializer().run()

  values.eval() ==> ["lake", "UNKNOWN"]
  ```

  Args:
    vocabulary_file: The vocabulary filename, may be a constant scalar `Tensor`.
    vocab_size: Number of the elements in the vocabulary, if known.
    default_value: The value to use for out-of-vocabulary indices.
    name: A name for this op (optional).
    key_column_index: The column index from the text file to get the `key`
      values from. The default is to use the line number, starting from zero.
    value_column_index: The column index from the text file to get the `value`
      values from. The default is to use the whole line content.
    delimiter: The delimiter to separate fields in a line.

  Returns:
    The lookup table to map a string values associated to a given index `int64`
    `Tensors`.

  Raises:
    ValueError: when `vocabulary_file` is empty.
    ValueError: when `vocab_size` is invalid.
  """
    if vocabulary_file is None or (isinstance(vocabulary_file, str) and (not vocabulary_file)):
        raise ValueError('`vocabulary_file` must be specified and must not be empty.')
    if vocab_size is not None and vocab_size < 1:
        raise ValueError(f'`vocab_size` must be greater than 0, got {vocab_size}.')
    with ops.name_scope(name, 'index_to_string'):
        init = TextFileStringTableInitializer(vocabulary_file, vocab_size=vocab_size, name='table_init', key_column_index=key_column_index, value_column_index=value_column_index, delimiter=delimiter)
        return StaticHashTableV1(init, default_value)