import inspect
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
def raise_zero_rank_error():
    msg = gen_string_ops.string_join(['len requires non-zero rank, got ', gen_string_ops.as_string(rank)])
    with ops.control_dependencies([control_flow_assert.Assert(False, [msg])]):
        return constant_op.constant(0, dtype=dtypes.int32)