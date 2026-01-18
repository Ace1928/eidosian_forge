from tensorflow.python.data.experimental.ops.cardinality import assert_cardinality
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
def initialize(self, table):
    lookup_ops.check_table_dtypes(table, self._key_dtype, self._value_dtype)
    init_op = ged_ops.initialize_table_from_dataset(table.resource_handle, self.dataset._variant_tensor)
    ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
    return init_op