import numpy as np
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@property
def nested_row_splits(self):
    """The row_splits for all ragged dimensions in this ragged tensor value."""
    rt_nested_splits = [self.row_splits]
    rt_values = self.values
    while isinstance(rt_values, RaggedTensorValue):
        rt_nested_splits.append(rt_values.row_splits)
        rt_values = rt_values.values
    return tuple(rt_nested_splits)