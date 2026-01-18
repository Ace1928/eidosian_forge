import collections
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import serialization
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('feature_column.sequence_categorical_column_with_identity')
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def sequence_categorical_column_with_identity(key, num_buckets, default_value=None):
    """Returns a feature column that represents sequences of integers.

  Pass this to `embedding_column` or `indicator_column` to convert sequence
  categorical data into dense representation for input to sequence NN, such as
  RNN.

  Example:

  ```python
  watches = sequence_categorical_column_with_identity(
      'watches', num_buckets=1000)
  watches_embedding = embedding_column(watches, dimension=10)
  columns = [watches_embedding]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
  ```

  Args:
    key: A unique string identifying the input feature.
    num_buckets: Range of inputs. Namely, inputs are expected to be in the range
      `[0, num_buckets)`.
    default_value: If `None`, this column's graph operations will fail for
      out-of-range inputs. Otherwise, this value must be in the range `[0,
      num_buckets)`, and will replace out-of-range inputs.

  Returns:
    A `SequenceCategoricalColumn`.

  Raises:
    ValueError: if `num_buckets` is less than one.
    ValueError: if `default_value` is not in range `[0, num_buckets)`.
  """
    return fc.SequenceCategoricalColumn(fc.categorical_column_with_identity(key=key, num_buckets=num_buckets, default_value=default_value))