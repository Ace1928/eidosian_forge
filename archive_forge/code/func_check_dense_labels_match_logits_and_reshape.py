from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.feature_column.feature_column import _NumericColumn
from tensorflow.python.framework import ops
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
def check_dense_labels_match_logits_and_reshape(labels, logits, expected_labels_dimension):
    """Checks labels shape matches logits, and reshapes if needed.

  Consider logits of shape [D0, D1, ... DN, logits_dimension]. Then labels
  shape must be [D0, D1, ... DN, expected_labels_dimension].
  If expected_labels_dimension=1, labels could be [D0, D1, ... DN] and this
  method reshapes them to [D0, D1, ... DN, 1].

  Args:
    labels: labels Tensor.
    logits: logits Tensor.
    expected_labels_dimension: Integer.

  Returns:
    Validated and reshaped labels Tensor.

  Raises:
    ValueError: If labels is a SparseTensor.
    ValueError: If labels shape is statically defined and fails validation.
    OpError: If labels shape is not statically defined and fails validation.
  """
    if labels is None:
        raise ValueError(_LABEL_NONE_ERR_MSG)
    with ops.name_scope('labels', values=(labels, logits)) as scope:
        labels = tf.compat.v1.convert_to_tensor_or_sparse_tensor(labels)
        if isinstance(labels, tf.sparse.SparseTensor):
            raise ValueError(_SPARSE_LABEL_ERR_MSG.format(expected_labels_dimension, expected_labels_dimension, expected_labels_dimension))
        if tf.executing_eagerly():
            labels_rank = labels._rank()
            logits_rank = logits._rank()
            if labels_rank is not None and logits_rank is not None and (labels_rank == logits_rank - 1):
                labels = tf.compat.v1.expand_dims(labels, -1)
                labels_rank += 1
            labels_shape = labels._shape_tuple()
            if labels_rank < 2:
                raise ValueError('labels must have rank at least 2.  Received rank {}, shape {}'.format(labels_rank, labels_shape))
            if labels_shape[-1] != expected_labels_dimension:
                raise ValueError(_MISMATCHED_LABEL_DIM_ERR_MSG.format(expected_labels_dimension, labels_shape[-1]))
            logits_shape = logits._shape_tuple()
            expected_labels_shape = logits_shape[:-1] + (expected_labels_dimension,)
            if expected_labels_shape != labels_shape:
                raise ValueError('{}, expected_labels_shape: {}. labels_shape: {}.'.format(_LABEL_SHAPE_ERR_MSG.format(expected_labels_dimension), expected_labels_shape, labels_shape))
            return labels
        if labels.shape.ndims is not None and logits.shape.ndims is not None and (labels.shape.ndims == logits.shape.ndims - 1):
            labels = tf.compat.v1.expand_dims(labels, -1)
        assert_rank = tf.compat.v1.debugging.assert_rank_at_least(labels, 2, message=_LABEL_SHAPE_ERR_MSG.format(expected_labels_dimension))
        with tf.control_dependencies([assert_rank]):
            static_shape = labels.shape
            if static_shape.ndims is not None:
                final_dim = static_shape[-1]
                if final_dim is not None and final_dim != expected_labels_dimension:
                    raise ValueError(_MISMATCHED_LABEL_DIM_ERR_MSG.format(expected_labels_dimension, final_dim))
            logits_shape = tf.compat.v1.shape(logits)
            expected_labels_shape = tf.concat([logits_shape[:-1], [expected_labels_dimension]], axis=0)
            labels_shape = tf.compat.v1.shape(labels)
            assert_dimension = tf.compat.v1.debugging.assert_equal(expected_labels_shape, labels_shape, message=_LABEL_SHAPE_ERR_MSG.format(expected_labels_dimension), data=['expected_labels_shape: ', expected_labels_shape, 'labels_shape: ', labels_shape])
            with tf.control_dependencies([assert_dimension]):
                return tf.identity(labels, name=scope)