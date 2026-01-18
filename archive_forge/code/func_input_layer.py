import abc
import collections
import math
import numpy as np
import six
from tensorflow.python.eager import context
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export(v1=['feature_column.input_layer'])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def input_layer(features, feature_columns, weight_collections=None, trainable=True, cols_to_vars=None, cols_to_output_tensors=None):
    """Returns a dense `Tensor` as input layer based on given `feature_columns`.

  Generally a single example in training data is described with FeatureColumns.
  At the first layer of the model, this column oriented data should be converted
  to a single `Tensor`.

  Example:

  ```python
  price = numeric_column('price')
  keywords_embedded = embedding_column(
      categorical_column_with_hash_bucket("keywords", 10K), dimensions=16)
  columns = [price, keywords_embedded, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  for units in [128, 64, 32]:
    dense_tensor = tf.compat.v1.layers.dense(dense_tensor, units, tf.nn.relu)
  prediction = tf.compat.v1.layers.dense(dense_tensor, 1)
  ```

  Args:
    features: A mapping from key to tensors. `_FeatureColumn`s look up via these
      keys. For example `numeric_column('price')` will look at 'price' key in
      this dict. Values can be a `SparseTensor` or a `Tensor` depends on
      corresponding `_FeatureColumn`.
    feature_columns: An iterable containing the FeatureColumns to use as inputs
      to your model. All items should be instances of classes derived from
      `_DenseColumn` such as `numeric_column`, `embedding_column`,
      `bucketized_column`, `indicator_column`. If you have categorical features,
      you can wrap them with an `embedding_column` or `indicator_column`.
    weight_collections: A list of collection names to which the Variable will be
      added. Note that variables will also be added to collections
      `tf.GraphKeys.GLOBAL_VARIABLES` and `ops.GraphKeys.MODEL_VARIABLES`.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    cols_to_vars: If not `None`, must be a dictionary that will be filled with a
      mapping from `_FeatureColumn` to list of `Variable`s.  For example, after
      the call, we might have cols_to_vars = {_EmbeddingColumn(
      categorical_column=_HashedCategoricalColumn( key='sparse_feature',
      hash_bucket_size=5, dtype=tf.string), dimension=10): [<tf.Variable
      'some_variable:0' shape=(5, 10), <tf.Variable 'some_variable:1' shape=(5,
      10)]} If a column creates no variables, its value will be an empty list.
    cols_to_output_tensors: If not `None`, must be a dictionary that will be
      filled with a mapping from '_FeatureColumn' to the associated output
      `Tensor`s.

  Returns:
    A `Tensor` which represents input layer of a model. Its shape
    is (batch_size, first_layer_dimension) and its dtype is `float32`.
    first_layer_dimension is determined based on given `feature_columns`.

  Raises:
    ValueError: if an item in `feature_columns` is not a `_DenseColumn`.
  """
    return _internal_input_layer(features, feature_columns, weight_collections=weight_collections, trainable=trainable, cols_to_vars=cols_to_vars, cols_to_output_tensors=cols_to_output_tensors)