import copy
import enum
import math
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.feature_column import _is_running_on_cpu
from tensorflow.python.tpu.feature_column import _record_variable_scope_and_name
from tensorflow.python.tpu.feature_column import _SUPPORTED_CATEGORICAL_COLUMNS_V2
from tensorflow.python.tpu.feature_column import _SUPPORTED_SEQUENCE_COLUMNS
from tensorflow.python.tpu.feature_column import _TPUBaseEmbeddingColumn
from tensorflow.python.util.tf_export import tf_export
def split_sequence_columns_v2(feature_columns):
    """Split a list of _TPUEmbeddingColumn into sequence and non-sequence columns.

  For use in a TPUEstimator model_fn function. E.g.

  def model_fn(features):
    sequence_columns, feature_columns = (
        tf.tpu.feature_column.split_sequence_columns(feature_columns))
    input = tf.feature_column.input_layer(
        features=features, feature_columns=feature_columns)
    sequence_features, sequence_lengths = (
        tf.contrib.feature_column.sequence_input_layer(
            features=features, feature_columns=sequence_columns))

  Args:
    feature_columns: A list of _TPUEmbeddingColumns to split.

  Returns:
    Two lists of _TPUEmbeddingColumns, the first is the sequence columns and the
    second is the non-sequence columns.
  """
    sequence_columns = []
    non_sequence_columns = []
    for column in feature_columns:
        if not isinstance(column, (_TPUEmbeddingColumnV2, _TPUSharedEmbeddingColumnV2)):
            raise TypeError(f'column must be a _TPUEmbeddingColumnV2 or _TPUSharedEmbeddingColumnV2 but got {type(column)} instead.')
        if column.is_sequence_column():
            sequence_columns.append(column)
        else:
            non_sequence_columns.append(column)
    return (sequence_columns, non_sequence_columns)