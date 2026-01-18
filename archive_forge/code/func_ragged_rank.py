import numpy as np
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@property
def ragged_rank(self):
    """The number of ragged dimensions in this ragged tensor value."""
    values_is_ragged = isinstance(self._values, RaggedTensorValue)
    return self._values.ragged_rank + 1 if values_is_ragged else 1