import abc
import collections
import math
import re
import numpy as np
import six
from tensorflow.python.data.experimental.ops import lookup_ops as data_lookup_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2_types as fc_types
from tensorflow.python.feature_column import serialization
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export(v1=['feature_column.categorical_column_with_vocabulary_file'])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def categorical_column_with_vocabulary_file(key, vocabulary_file, vocabulary_size=None, num_oov_buckets=0, default_value=None, dtype=dtypes.string):
    """A `CategoricalColumn` with a vocabulary file.

  Use this when your inputs are in string or integer format, and you have a
  vocabulary file that maps each value to an integer ID. By default,
  out-of-vocabulary values are ignored. Use either (but not both) of
  `num_oov_buckets` and `default_value` to specify how to include
  out-of-vocabulary values.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example with `num_oov_buckets`:
  File '/us/states.txt' contains 50 lines, each with a 2-character U.S. state
  abbreviation. All inputs with values in that file are assigned an ID 0-49,
  corresponding to its line number. All other values are hashed and assigned an
  ID 50-54.

  ```python
  import tensorflow as tf
  states = tf.feature_column.categorical_column_with_vocabulary_file(
    key='states', vocabulary_file='states.txt', vocabulary_size=5,
    num_oov_buckets=1)
  columns = [states]
  features = {'states':tf.constant([['california', 'georgia', 'michigan',
  'texas', 'new york'], ['new york', 'georgia', 'california', 'michigan',
  'texas']])}
  linear_prediction = tf.compat.v1.feature_column.linear_model(features,
  columns)
  ```

  Example with `default_value`:
  File '/us/states.txt' contains 51 lines - the first line is 'XX', and the
  other 50 each have a 2-character U.S. state abbreviation. Both a literal 'XX'
  in input, and other values missing from the file, will be assigned ID 0. All
  others are assigned the corresponding line number 1-50.

  ```python
  import tensorflow as tf
  states = tf.feature_column.categorical_column_with_vocabulary_file(
    key='states', vocabulary_file='states.txt', vocabulary_size=6,
    default_value=0)
  columns = [states]
  features = {'states':tf.constant([['california', 'georgia', 'michigan',
  'texas', 'new york'], ['new york', 'georgia', 'california', 'michigan',
  'texas']])}
  linear_prediction = tf.compat.v1.feature_column.linear_model(features,
  columns)
  ```

  And to make an embedding with either:

  ```python
  import tensorflow as tf
  states = tf.feature_column.categorical_column_with_vocabulary_file(
    key='states', vocabulary_file='states.txt', vocabulary_size=5,
    num_oov_buckets=1)
  columns = [tf.feature_column.embedding_column(states, 3)]
  features = {'states':tf.constant([['california', 'georgia', 'michigan',
  'texas', 'new york'], ['new york', 'georgia', 'california', 'michigan',
  'texas']])}
  input_layer = tf.keras.layers.DenseFeatures(columns)
  dense_tensor = input_layer(features)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the column
      name and the dictionary key for feature parsing configs, feature `Tensor`
      objects, and feature columns.
    vocabulary_file: The vocabulary file name.
    vocabulary_size: Number of the elements in the vocabulary. This must be no
      greater than length of `vocabulary_file`, if less than length, later
      values are ignored. If None, it is set to the length of `vocabulary_file`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of
      the input value. A positive `num_oov_buckets` can not be specified with
      `default_value`.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `CategoricalColumn` with a vocabulary file.

  Raises:
    ValueError: `vocabulary_file` is missing or cannot be opened.
    ValueError: `vocabulary_size` is missing or < 1.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: `dtype` is neither string nor integer.
  """
    return categorical_column_with_vocabulary_file_v2(key, vocabulary_file, vocabulary_size, dtype, default_value, num_oov_buckets)