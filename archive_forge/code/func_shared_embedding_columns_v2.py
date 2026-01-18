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
@tf_export('feature_column.shared_embeddings', v1=[])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def shared_embedding_columns_v2(categorical_columns, dimension, combiner='mean', initializer=None, shared_embedding_collection_name=None, ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None, trainable=True, use_safe_embedding_lookup=True):
    """List of dense columns that convert from sparse, categorical input.

  This is similar to `embedding_column`, except that it produces a list of
  embedding columns that share the same embedding weights.

  Use this when your inputs are sparse and of the same type (e.g. watched and
  impression video IDs that share the same vocabulary), and you want to convert
  them to a dense representation (e.g., to feed to a DNN).

  Inputs must be a list of categorical columns created by any of the
  `categorical_column_*` function. They must all be of the same type and have
  the same arguments except `key`. E.g. they can be
  categorical_column_with_vocabulary_file with the same vocabulary_file. Some or
  all columns could also be weighted_categorical_column.

  Here is an example embedding of two features for a DNNClassifier model:

  ```python
  watched_video_id = categorical_column_with_vocabulary_file(
      'watched_video_id', video_vocabulary_file, video_vocabulary_size)
  impression_video_id = categorical_column_with_vocabulary_file(
      'impression_video_id', video_vocabulary_file, video_vocabulary_size)
  columns = shared_embedding_columns(
      [watched_video_id, impression_video_id], dimension=10)

  estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)

  label_column = ...
  def input_fn():
    features = tf.io.parse_example(
        ..., features=make_parse_example_spec(columns + [label_column]))
    labels = features.pop(label_column.name)
    return features, labels

  estimator.train(input_fn=input_fn, steps=100)
  ```

  Here is an example using `shared_embedding_columns` with model_fn:

  ```python
  def model_fn(features, ...):
    watched_video_id = categorical_column_with_vocabulary_file(
        'watched_video_id', video_vocabulary_file, video_vocabulary_size)
    impression_video_id = categorical_column_with_vocabulary_file(
        'impression_video_id', video_vocabulary_file, video_vocabulary_size)
    columns = shared_embedding_columns(
        [watched_video_id, impression_video_id], dimension=10)
    dense_tensor = input_layer(features, columns)
    # Form DNN layers, calculate loss, and return EstimatorSpec.
    ...
  ```

  Args:
    categorical_columns: List of categorical columns created by a
      `categorical_column_with_*` function. These columns produce the sparse IDs
      that are inputs to the embedding lookup. All columns must be of the same
      type and have the same arguments except `key`. E.g. they can be
      categorical_column_with_vocabulary_file with the same vocabulary_file.
      Some or all columns could also be weighted_categorical_column.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional collective name of these columns.
      If not given, a reasonable name will be chosen based on the names of
      `categorical_columns`.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from which
      to restore the column weights. Required if `ckpt_to_load_from` is not
      `None`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    trainable: Whether or not the embedding is trainable. Default is True.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies to rank 2 (NxM) shaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.

  Returns:
    A list of dense columns that converts from sparse input. The order of
    results follows the ordering of `categorical_columns`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if any of the given `categorical_columns` is of different type
      or has different arguments than the others.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: if eager execution is enabled.
  """
    if context.executing_eagerly():
        raise RuntimeError('shared_embedding_columns are not supported when eager execution is enabled.')
    if dimension is None or dimension < 1:
        raise ValueError('Invalid dimension {}.'.format(dimension))
    if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
        raise ValueError('Must specify both `ckpt_to_load_from` and `tensor_name_in_ckpt` or none of them.')
    if initializer is not None and (not callable(initializer)):
        raise ValueError('initializer must be callable if specified.')
    if initializer is None:
        initializer = init_ops.truncated_normal_initializer(mean=0.0, stddev=1.0 / math.sqrt(dimension))
    sorted_columns = sorted(categorical_columns, key=lambda x: x.name)
    c0 = sorted_columns[0]
    num_buckets = c0.num_buckets
    if not isinstance(c0, CategoricalColumn):
        raise ValueError('All categorical_columns must be subclasses of CategoricalColumn. Given: {}, of type: {}'.format(c0, type(c0)))
    while isinstance(c0, (WeightedCategoricalColumn, SequenceCategoricalColumn)):
        c0 = c0.categorical_column
    for c in sorted_columns[1:]:
        while isinstance(c, (WeightedCategoricalColumn, SequenceCategoricalColumn)):
            c = c.categorical_column
        if not isinstance(c, type(c0)):
            raise ValueError('To use shared_embedding_column, all categorical_columns must have the same type, or be weighted_categorical_column or sequence column of the same type. Given column: {} of type: {} does not match given column: {} of type: {}'.format(c0, type(c0), c, type(c)))
        if num_buckets != c.num_buckets:
            raise ValueError('To use shared_embedding_column, all categorical_columns must have the same number of buckets. Given column: {} with buckets: {} does  not match column: {} with buckets: {}'.format(c0, num_buckets, c, c.num_buckets))
    if not shared_embedding_collection_name:
        shared_embedding_collection_name = '_'.join((c.name for c in sorted_columns))
        shared_embedding_collection_name += '_shared_embedding'
    column_creator = SharedEmbeddingColumnCreator(dimension, initializer, ckpt_to_load_from, tensor_name_in_ckpt, num_buckets, trainable, shared_embedding_collection_name, use_safe_embedding_lookup)
    result = []
    for column in categorical_columns:
        result.append(column_creator(categorical_column=column, combiner=combiner, max_norm=max_norm))
    return result