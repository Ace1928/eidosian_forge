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
@tf_export('feature_column.bucketized_column')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def bucketized_column(source_column, boundaries):
    """Represents discretized dense input bucketed by `boundaries`.

  Buckets include the left boundary, and exclude the right boundary. Namely,
  `boundaries=[0., 1., 2.]` generates buckets `(-inf, 0.)`, `[0., 1.)`,
  `[1., 2.)`, and `[2., +inf)`.

  For example, if the inputs are

  ```python
  boundaries = [0, 10, 100]
  input tensor = [[-5, 10000]
                  [150,   10]
                  [5,    100]]
  ```

  then the output will be

  ```python
  output = [[0, 3]
            [3, 2]
            [1, 3]]
  ```

  Example:

  ```python
  price = tf.feature_column.numeric_column('price')
  bucketized_price = tf.feature_column.bucketized_column(
      price, boundaries=[...])
  columns = [bucketized_price, ...]
  features = tf.io.parse_example(
      ..., features=tf.feature_column.make_parse_example_spec(columns))
  dense_tensor = tf.keras.layers.DenseFeatures(columns)(features)
  ```

  A `bucketized_column` can also be crossed with another categorical column
  using `crossed_column`:

  ```python
  price = tf.feature_column.numeric_column('price')
  # bucketized_column converts numerical feature to a categorical one.
  bucketized_price = tf.feature_column.bucketized_column(
      price, boundaries=[...])
  # 'keywords' is a string feature.
  price_x_keywords = tf.feature_column.crossed_column(
      [bucketized_price, 'keywords'], 50K)
  columns = [price_x_keywords, ...]
  features = tf.io.parse_example(
      ..., features=tf.feature_column.make_parse_example_spec(columns))
  dense_tensor = tf.keras.layers.DenseFeatures(columns)(features)
  linear_model = tf.keras.experimental.LinearModel(units=...)(dense_tensor)
  ```

  Args:
    source_column: A one-dimensional dense column which is generated with
      `numeric_column`.
    boundaries: A sorted list or tuple of floats specifying the boundaries.

  Returns:
    A `BucketizedColumn`.

  Raises:
    ValueError: If `source_column` is not a numeric column, or if it is not
      one-dimensional.
    ValueError: If `boundaries` is not a sorted list or tuple.
  """
    if not isinstance(source_column, (NumericColumn, fc_old._NumericColumn)):
        raise ValueError('source_column must be a column generated with numeric_column(). Given: {}'.format(source_column))
    if len(source_column.shape) > 1:
        raise ValueError('source_column must be one-dimensional column. Given: {}'.format(source_column))
    if not boundaries:
        raise ValueError('boundaries must not be empty.')
    if not (isinstance(boundaries, list) or isinstance(boundaries, tuple)):
        raise ValueError('boundaries must be a sorted list.')
    for i in range(len(boundaries) - 1):
        if boundaries[i] >= boundaries[i + 1]:
            raise ValueError('boundaries must be a sorted list.')
    return BucketizedColumn(source_column, tuple(boundaries))