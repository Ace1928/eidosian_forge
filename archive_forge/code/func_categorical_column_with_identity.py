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
@tf_export('feature_column.categorical_column_with_identity')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def categorical_column_with_identity(key, num_buckets, default_value=None):
    """A `CategoricalColumn` that returns identity values.

  Use this when your inputs are integers in the range `[0, num_buckets)`, and
  you want to use the input value itself as the categorical ID. Values outside
  this range will result in `default_value` if specified, otherwise it will
  fail.

  Typically, this is used for contiguous ranges of integer indexes, but
  it doesn't have to be. This might be inefficient, however, if many of IDs
  are unused. Consider `categorical_column_with_hash_bucket` in that case.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  In the following examples, each input in the range `[0, 1000000)` is assigned
  the same value. All other inputs are assigned `default_value` 0. Note that a
  literal 0 in inputs will result in the same default ID.

  Linear model:

  ```python
  import tensorflow as tf
  video_id = tf.feature_column.categorical_column_with_identity(
      key='video_id', num_buckets=1000000, default_value=0)
  columns = [video_id]
  features = {'video_id': tf.sparse.from_dense([[2, 85, 0, 0, 0],
  [33,78, 2, 73, 1]])}
  linear_prediction = tf.compat.v1.feature_column.linear_model(features,
  columns)
  ```

  Embedding for a DNN model:

  ```python
  import tensorflow as tf
  video_id = tf.feature_column.categorical_column_with_identity(
      key='video_id', num_buckets=1000000, default_value=0)
  columns = [tf.feature_column.embedding_column(video_id, 9)]
  features = {'video_id': tf.sparse.from_dense([[2, 85, 0, 0, 0],
  [33,78, 2, 73, 1]])}
  input_layer = tf.keras.layers.DenseFeatures(columns)
  dense_tensor = input_layer(features)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the column
      name and the dictionary key for feature parsing configs, feature `Tensor`
      objects, and feature columns.
    num_buckets: Range of inputs and outputs is `[0, num_buckets)`.
    default_value: If set, values outside of range `[0, num_buckets)` will be
      replaced with this value. If not set, values >= num_buckets will cause a
      failure while values < 0 will be dropped.

  Returns:
    A `CategoricalColumn` that returns identity values.

  Raises:
    ValueError: if `num_buckets` is less than one.
    ValueError: if `default_value` is not in range `[0, num_buckets)`.
  """
    if num_buckets < 1:
        raise ValueError('num_buckets {} < 1, column_name {}'.format(num_buckets, key))
    if default_value is not None and (default_value < 0 or default_value >= num_buckets):
        raise ValueError('default_value {} not in range [0, {}), column_name {}'.format(default_value, num_buckets, key))
    fc_utils.assert_key_is_string(key)
    return IdentityCategoricalColumn(key=key, number_buckets=num_buckets, default_value=default_value)