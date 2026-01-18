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
def get_sequence_dense_tensor(self, transformation_cache, state_manager):
    """Returns a `TensorSequenceLengthPair`.

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.
    """
    sp_tensor = transformation_cache.get(self, state_manager)
    dense_tensor = sparse_ops.sparse_tensor_to_dense(sp_tensor, default_value=self.default_value)
    dense_shape = array_ops.concat([array_ops.shape(dense_tensor)[:1], [-1], self.variable_shape], axis=0)
    dense_tensor = array_ops.reshape(dense_tensor, shape=dense_shape)
    if sp_tensor.shape.ndims == 2:
        num_elements = self.variable_shape.num_elements()
    else:
        num_elements = 1
    seq_length = fc_utils.sequence_length_from_sparse_tensor(sp_tensor, num_elements=num_elements)
    return fc.SequenceDenseColumn.TensorSequenceLengthPair(dense_tensor=dense_tensor, sequence_length=seq_length)