from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def register_overrides():
    """Registers the autograph specific overrides for dataset_ops."""
    control_flow.for_loop_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_for_stmt)
    py_builtins.abs_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_abs)
    py_builtins.len_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_len)
    py_builtins.enumerate_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_enumerate)
    py_builtins.zip_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_zip)
    py_builtins.map_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_map)
    py_builtins.filter_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_filter)
    py_builtins.any_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_any)
    py_builtins.all_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_all)